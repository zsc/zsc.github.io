# train.py
"""
Implements prompt-tuning for Qwen2.5-1.5B-Instruct on the GSM8K dataset.

This script provides the core logic for training a soft prompt that can be used to
enhance the model's mathematical reasoning capabilities. It is designed to be
controlled by a Flask application (app.py) but can also be run standalone.

Note on Implementation Choice:
The project is named SLOT, after the paper "Sample-specific Language Model
Optimization at Test-time". However, the SLOT paper proposes an *inference-time*
optimization technique where a small parameter vector is tuned for each individual
sample and then discarded. This does not involve a persistent training phase or
checkpoints in the traditional sense.

To meet the user's requirement for a web UI with "train" and "test" tabs,
start/stop functionality, and checkpoints, this script implements a more
conventional *prompt-tuning* approach. A single soft prompt (a small set of
learnable vectors) is trained on the entire GSM8K training set. This trained
prompt is saved as a checkpoint and can be loaded for evaluation. This approach
is more aligned with a standard training workflow while still capturing the spirit
of adapting the model with a small number of parameters.
"""
import os
import re
import sys
import time
import json
import logging
import unittest
import argparse
from datetime import datetime
from threading import Event
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.attention import SDPBackend, sdpa_kernel

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "gsm8k"
DATASET_CONFIG = "main"

# --- Utility Functions ---

def extract_final_answer(text: str):
    """
    Extracts the final numerical answer from a GSM8K-style response.
    The answer is typically the last number in the string, often after '####'.
    """
    # Find all numbers in the string, including integers and floats
    # This regex handles negative numbers, and numbers with commas
    text = text.replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)

    if not numbers:
        return None

    try:
        # The answer is the last number found in the text
        return float(numbers[-1])
    except (ValueError, IndexError):
        return None

# --- Core Model and Training Logic ---

class PromptTuningModel(nn.Module):
    """
    A wrapper for a frozen LLM with a learnable soft prompt.
    """
    def __init__(self, model_name: str, num_prompt_tokens: int, device: torch.device):
        super().__init__()
        self.device = device
        self.num_prompt_tokens = num_prompt_tokens

        logging.info(f"Loading base model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)

        # Freeze all parameters of the base model
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = self.model.config.hidden_size

        # Initialize the soft prompt
        self.soft_prompt = nn.Parameter(
            torch.randn(1, self.num_prompt_tokens, self.hidden_size, dtype=torch.bfloat16, device=device)
        )
        logging.info(f"Initialized soft prompt with shape: {self.soft_prompt.shape}")

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Get word embeddings for the input_ids
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        batch_size = inputs_embeds.size(0)

        # 2. Prepend the soft prompt embeddings
        # Expand soft_prompt to match the batch size
        prompt_embeds = self.soft_prompt.expand(batch_size, -1, -1)
        full_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        # 3. Adjust attention mask for the prepended prompt
        prompt_attention_mask = torch.ones(
            batch_size, self.num_prompt_tokens, dtype=attention_mask.dtype, device=self.device
        )
        full_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

        # 4. Adjust labels to ignore loss on prompt tokens
        if labels is not None:
            prompt_labels = torch.full(
                (batch_size, self.num_prompt_tokens), -100, dtype=labels.dtype, device=self.device
            )
            full_labels = torch.cat([prompt_labels, labels], dim=1)
        else:
            full_labels = None

        # 5. Pass everything to the base model
        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
        return outputs

def train(args: argparse.Namespace, stop_event: Event, log_queue=None):
    """
    Main training function, designed to be run in a separate process.
    """
    def log(message):
        logging.info(message)
        if log_queue:
            log_queue.put(message)

    try:
        # --- Setup ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            log("WARNING: CUDA not available, training on CPU. This will be very slow.")

        # Create unique experiment name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"SLOT_Qwen2.5-1.5B_{args.num_prompt_tokens}tokens_lr{args.learning_rate}_bs{args.batch_size}_{timestamp}"
        
        checkpoints_dir = Path(args.checkpoints_dir) / exp_name
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        log(f"Experiment: {exp_name}")
        log(f"Checkpoints will be saved to: {checkpoints_dir}")

        writer = SummaryWriter(log_dir=f'runs/{exp_name}')

        # --- Model and Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = PromptTuningModel(MODEL_ID, args.num_prompt_tokens, device)

        # --- Data ---
        log("Loading and preparing GSM8K dataset...")
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
        
        def format_and_tokenize(examples):
            questions = [f"Question: {q}\nAnswer: " for q in examples['question']]
            full_texts = [q + a for q, a in zip(questions, examples['answer'])]
            
            # Tokenize the full text
            tokenized_full = tokenizer(full_texts, truncation=True, max_length=args.max_length, padding="max_length")
            # Tokenize just the question part to identify where the answer starts
            tokenized_questions = tokenizer(questions, add_special_tokens=False)

            labels = torch.tensor(tokenized_full['input_ids'])
            
            # Mask out the question part from the labels by setting them to -100
            for i in range(len(labels)):
                question_len = len(tokenized_questions['input_ids'][i])
                labels[i, :question_len] = -100

            tokenized_full['labels'] = labels.tolist()
            return tokenized_full

        train_dataset = dataset['train'].select(range(args.num_train_samples)).map(format_and_tokenize, batched=True)
        test_dataset = dataset['test'].select(range(args.num_test_samples))

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # --- Optimizer and Scheduler ---
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        num_training_steps = args.epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        # --- Checkpoint Loading ---
        start_epoch = 0
        global_step = 0
        if args.checkpoint_path:
            log(f"Resuming from checkpoint: {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.soft_prompt.data = checkpoint['soft_prompt']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            log(f"Resumed from epoch {start_epoch}, global step {global_step}")

        # --- Training Loop ---
        log("Starting training...")
        model.train()

        # Define some fixed prompts for qualitative evaluation on TensorBoard
        sample_generation_prompts = [
            {
                "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "answer": "Natalia sold 48/2 = 24 clips in May. In total, she sold 48 + 24 = 72 clips. #### 72"
            },
            {
                "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                "answer": "Weng earns 12/60 = $0.2 per minute. So she earned 0.2 * 50 = $10. #### 10"
            },
            {
                "question": "A deep-sea monster rises from the depths at a rate of 25 feet per minute. It is first spotted at a depth of 3000 feet. At the same time, a diving bell is 500 feet below sea level and is descending at a rate of 15 feet per minute. How long does it take for the monster to reach the diving bell?",
                "answer": "The monster is at -3000 feet and the diving bell is at -500 feet. The initial distance between them is 3000 - 500 = 2500 feet. The monster is rising at 25 ft/min and the bell is descending at 15 ft/min, so their relative speed of approach is 25 + 15 = 40 feet per minute. The time it takes for them to meet is 2500 feet / 40 feet/minute = 62.5 minutes. #### 62.5"
            }
        ]

        for epoch in range(start_epoch, args.epochs):
            if stop_event.is_set():
                log("Training stopped by user.")
                break

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                if stop_event.is_set():
                    break
                
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('LearningRate', lr_scheduler.get_last_lr()[0], global_step)
                global_step += 1

                # Periodic checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = checkpoints_dir / f"checkpoint_step_{global_step}.pt"
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'soft_prompt': model.soft_prompt.data,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                    }, checkpoint_path)
                    log(f"Saved checkpoint to {checkpoint_path}")

                # Log sample generations to TensorBoard periodically
                if global_step % 3000 == 0:
                    log("Generating sample outputs for TensorBoard...")
                    model.eval()
                    with torch.no_grad():
                        base_model = model.model
                        prompt_embeds = model.soft_prompt.to(device)

                        for i, prompt_data in enumerate(sample_generation_prompts):
                            question_text = f"Question: {prompt_data['question']}\nAnswer:"
                            inputs = tokenizer(question_text, return_tensors='pt').to(device)
                            input_ids = inputs.input_ids

                            inputs_embeds = base_model.get_input_embeddings()(input_ids)
                            full_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

                            outputs = base_model.generate(
                                inputs_embeds=full_embeds,
                                max_new_tokens=args.max_new_tokens,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                do_sample=False
                            )

                            generated_ids = outputs[0][input_ids.shape[1]:]
                            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                            log_text = (f"**Question:**\n{prompt_data['question']}\n\n"
                                        f"**Generated Answer:**\n```\n{generated_text}\n```\n\n"
                                        f"**Ground Truth:**\n```\n{prompt_data['answer']}\n```")
                            writer.add_text(f'SampleGeneration/{i+1}', log_text, global_step)
                    model.train()
                    log("Finished generating samples.")


            # Run evaluation at the end of each epoch
            if not stop_event.is_set():
                log(f"Starting evaluation for epoch {epoch+1}...")
                current_prompt = model.soft_prompt.data.clone()
                # We need to create a temporary argparse object for evaluation
                eval_args = argparse.Namespace(
                    checkpoint_path=None,  # Not needed, we pass prompt directly
                    num_test_samples=args.num_test_samples,
                    batch_size=args.batch_size,
                    max_new_tokens=256
                )
                accuracy = evaluate(eval_args, prompt_tensor=current_prompt, log_queue=log_queue)
                log(f"Epoch {epoch+1} Test Accuracy: {accuracy:.4f}")
                writer.add_scalar('Accuracy/test', accuracy, global_step)
                model.train() # Set model back to training mode

        log("Training finished.")

    except Exception as e:
        log(f"An error occurred during training: {e}")
        import traceback
        log(traceback.format_exc())
    finally:
        if 'writer' in locals() and writer:
            writer.close()
        log("Training process terminated.")

def evaluate(args: argparse.Namespace, prompt_tensor=None, log_queue=None):
    """
    Evaluation function. Uses Flash Attention 2 for generation.
    Can be called with a checkpoint path or a direct prompt tensor.
    """
    def log(message):
        logging.info(message)
        if log_queue:
            log_queue.put(message)
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Model and Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load base model, not the prompt-tuning wrapper
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.eval()

    if prompt_tensor is None:
        if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
            log(f"Error: Checkpoint path not provided or does not exist: {args.checkpoint_path}")
            return 0.0
        log(f"Loading soft prompt from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        prompt_tensor = checkpoint['soft_prompt']
    
    num_prompt_tokens = prompt_tensor.shape[1]
    prompt_embeds = prompt_tensor.to(device)

    # --- Data ---
    log("Loading GSM8K test set...")
    test_dataset = load_dataset(DATASET_ID, DATASET_CONFIG)['test'].select(range(args.num_test_samples))

    correct_predictions = 0
    total_predictions = 0

    pbar = tqdm(test_dataset, desc="Evaluating")
    for sample in pbar:
        question_text = f"Question: {sample['question']}\nAnswer:"
        
        # Tokenize question
        inputs = tokenizer(question_text, return_tensors='pt').to(device)
        input_ids = inputs.input_ids
        
        # Prepend prompt embeddings
        inputs_embeds = base_model.get_input_embeddings()(input_ids)
        full_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        # Generate using flash attention
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                outputs = base_model.generate(
                    inputs_embeds=full_embeds,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False # Use greedy decoding for determinism
                )
        except (ImportError, RuntimeError): # Fallback if flash attention not available
            log("Flash Attention 2 not available. Using default attention.")
            outputs = base_model.generate(
                inputs_embeds=full_embeds,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

        # Decode and extract answer
        # We need to slice off the prompt part from the output generation
        # The generated output contains the input_ids, so we slice from the end of the original input_ids
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        predicted_answer = extract_final_answer(generated_text)
        true_answer = extract_final_answer(sample['answer'])

        if predicted_answer is not None and true_answer is not None and abs(predicted_answer - true_answer) < 1e-3:
            correct_predictions += 1
        
        total_predictions += 1
        pbar.set_postfix({"Accuracy": f"{correct_predictions/total_predictions:.4f}"})

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    log(f"Final Test Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    return accuracy

# --- Unit Tests ---

class TestGsm8k(unittest.TestCase):
    def test_answer_extraction(self):
        self.assertEqual(extract_final_answer("The answer is 1,234."), 1234.0)
        self.assertEqual(extract_final_answer("So the result is 50."), 50.0)
        self.assertEqual(extract_final_answer("I think it is -10.5."), -10.5)
        self.assertEqual(extract_final_answer("Final Answer: \n#### 300"), 300.0)
        self.assertIsNone(extract_final_answer("There is no number here."))
        self.assertEqual(extract_final_answer("She had 10 apples and gave away 4, so she has 6 left."), 6.0)

    def test_model_initialization_and_forward(self):
        device = torch.device("cpu") # Use CPU for lightweight testing
        model = PromptTuningModel(MODEL_ID, num_prompt_tokens=10, device=device)

        # Check if base model is frozen
        for param in model.model.parameters():
            self.assertFalse(param.requires_grad)

        # Check if soft prompt is learnable
        self.assertTrue(model.soft_prompt.requires_grad)

        # Test forward pass
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        input_text = ["This is a test."]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        labels = inputs.input_ids.clone()

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                labels=labels.to(device)
            )
        self.assertIn('loss', outputs)
        self.assertIsNotNone(outputs.loss)
        self.assertIn('logits', outputs)
        
        # Check that logits shape is correct (batch, seq_len + prompt_len, vocab_size)
        expected_seq_len = inputs.input_ids.shape[1] + 10
        self.assertEqual(outputs.logits.shape, (1, expected_seq_len, model.model.config.vocab_size))

# --- Main execution block for standalone running ---

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a prompt-tuning model.")
    parser.add_argument('mode', choices=['train', 'evaluate'], help="Mode to run the script in.")
    
    # Training args
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=2, help="Batch size for training.")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument('--weight-decay', type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument('--num-prompt-tokens', type=int, default=20, help="Number of tokens in the soft prompt.")
    parser.add_argument('--max-length', type=int, default=512, help="Max sequence length for tokenization.")
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help="Directory to save checkpoints.")
    parser.add_argument('--save-steps', type=int, default=500, help="Save a checkpoint every N steps.")
    parser.add_argument('--num-train-samples', type=int, default=200, help="Number of training samples to use.")
    #parser.add_argument('--log-generation-steps', type=int, default=3000, help="Log sample generations to TensorBoard every N steps.")
    parser.add_argument('--num-test-samples', type=int, default=200, help="Number of test samples to use for evaluation.")
    
    # Common args
    parser.add_argument('--checkpoint-path', type=str, default=None, help="Path to a checkpoint to resume training from or for evaluation.")
    
    # Evaluation args
    parser.add_argument('--max-new-tokens', type=int, default=256, help="Max new tokens for generation during evaluation.")
    
    args = parser.parse_args()

    if args.mode == 'train':
        # For standalone training, create a dummy stop event that is never set
        stop_event = Event()
        train(args, stop_event)
    elif args.mode == 'evaluate':
        if not args.checkpoint_path:
            print("Error: --checkpoint-path is required for evaluation mode.")
            sys.exit(1)
        evaluate(args)

if __name__ == '__main__':
    # If 'unittest' is in the command-line arguments, run the tests
    if 'unittest' in sys.argv:
        # We need to remove 'unittest' and other unittest-related args from sys.argv
        # to prevent argparse from trying to parse them.
        sys.argv = [sys.argv[0]]
        unittest.main()
    else:
        main()
