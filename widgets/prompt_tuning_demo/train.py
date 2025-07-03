# train.py

import os
import re
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Any

# --- Configuration ---
class TrainConfig:
    # Model and Data
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    DATASET_ID = "gsm8k"
    DATASET_SPLIT = "test"
    NUM_TEST_SAMPLES = 200
    
    # Prompt Tuning Hyperparameters
    NUM_PROMPT_TOKENS = 20
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 3
    BATCH_SIZE = 2 # Keep it low due to long sequences
    
    # Generation Parameters for Evaluation
    MAX_NEW_TOKENS = 512
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16
    
    # Paths
    BASE_DIR = "experiments"
    
    def to_dict(self):
        return {
            "model_id": self.MODEL_ID,
            "num_prompt_tokens": self.NUM_PROMPT_TOKENS,
            "lr": self.LEARNING_RATE,
            "epochs": self.NUM_EPOCHS,
            "batch_size": self.BATCH_SIZE
        }

# --- Prompt Tuning Model Wrapper ---
class PromptTunedModel(nn.Module):
    """
    A wrapper class to implement prompt tuning manually.
    It adds a trainable soft prompt to the input embeddings of a frozen LLM.
    """
    def __init__(self, model: PreTrainedModel, num_prompt_tokens: int):
        super().__init__()
        self.model = model
        self.num_prompt_tokens = num_prompt_tokens

        # Freeze all original model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            param.data = param.data.to(TrainConfig.DTYPE)

        # Create the trainable soft prompt
        embedding_dim = self.model.config.hidden_size
        self.soft_prompt = nn.Parameter(
            torch.randn(1, self.num_prompt_tokens, embedding_dim, dtype=TrainConfig.DTYPE)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass that prepends the soft prompt to the input embeddings.
        """
        batch_size = input_ids.shape[0]
        
        # Get embeddings for tokenized input
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Prepend soft prompt embeddings
        # soft_prompt_embeds shape: [1, num_prompt_tokens, hidden_size]
        # inputs_embeds shape: [batch_size, seq_len, hidden_size]
        # We need to expand the soft prompt to match the batch size
        expanded_soft_prompt = self.soft_prompt.expand(batch_size, -1, -1)
        full_embeds = torch.cat([expanded_soft_prompt, inputs_embeds], dim=1)

        # Adjust attention mask for the new prompt tokens
        prompt_mask = torch.ones(
            batch_size, self.num_prompt_tokens, dtype=attention_mask.dtype, device=self.model.device
        )
        full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Adjust labels for the new prompt tokens (ignore them in loss calculation)
        if labels is not None:
            prompt_labels = torch.full(
                (batch_size, self.num_prompt_tokens), -100, dtype=labels.dtype, device=self.model.device
            )
            full_labels = torch.cat([prompt_labels, labels], dim=1)
        else:
            full_labels = None

        # Pass inputs_embeds instead of input_ids to the base model
        return self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """
        Generation method for the prompt-tuned model.
        """
        batch_size = input_ids.shape[0]
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        expanded_soft_prompt = self.soft_prompt.expand(batch_size, -1, -1)
        full_embeds = torch.cat([expanded_soft_prompt, inputs_embeds], dim=1)

        prompt_mask = torch.ones(
            batch_size, self.num_prompt_tokens, dtype=attention_mask.dtype, device=self.model.device
        )
        full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # We must use inputs_embeds for generation.
        # --- START OF FIX ---
        # CRUCIAL FIX: We must also pass the original `input_ids`.
        # The `generate` method uses `input_ids` to construct the full output sequence,
        # ensuring the prompt text is included before the newly generated text.
        # While the model's forward pass will use `inputs_embeds` for computation,
        # `input_ids` is essential for shaping the final output token sequence.
        # Without it, `generate` might only return the new tokens, leading to
        # incomplete or incorrect decoding, as observed in the original problem.
        return self.model.generate(
            input_ids=input_ids,
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            **kwargs
        )
        # --- END OF FIX ---


# --- Data and Evaluation Utilities ---
def extract_answer(text: str) -> Optional[str]:
    """Extracts the final numeric answer from a GSM8K-formatted string."""
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1).strip()
    # Fallback for simple numeric answers
    text = text.split("The final answer is")[-1]
    results = re.findall(r"(-?\d+\.?\d+)", text)
    if results:
        return results[-1]
    return None

def prepare_data(tokenizer: PreTrainedTokenizer, num_samples: int):
    """Loads and preprocesses the GSM8K dataset."""
    dataset = load_dataset(TrainConfig.DATASET_ID, "main")[TrainConfig.DATASET_SPLIT]
    dataset = dataset.select(range(num_samples))

    def format_example(example):
        # Apply the chat template format for Qwen2
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves math problems."},
            {"role": "user", "content": f"Question: {example['question']}\nThink step-by-step and then write the final answer in the format #### <number>."}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Adds the assistant role token
        )
        # The answer part includes the thinking process and the final answer
        full_response = example['answer']
        return {"prompt": prompt, "full_response": full_response}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    def tokenize_function(examples):
        # Tokenize prompt and response separately
        inputs = tokenizer(examples["prompt"], truncation=False, padding=False)
        targets = tokenizer(examples["full_response"], truncation=False, padding=False)

        # Concatenate for labels, then apply mask
        input_ids = []
        labels = []
        attention_mask = []
        
        for i in range(len(inputs.input_ids)):
            input_len = len(inputs.input_ids[i])
            
            # Combine prompt and response tokens
            combined_ids = inputs.input_ids[i] + targets.input_ids[i] + [tokenizer.eos_token_id]
            
            # Create labels: mask out the prompt part by setting it to -100
            label = ([-100] * input_len) + targets.input_ids[i] + [tokenizer.eos_token_id]
            
            input_ids.append(torch.tensor(combined_ids))
            labels.append(torch.tensor(label))
            attention_mask.append(torch.ones(len(combined_ids)))
        
        # We will pad later in the data loader
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    
    # We can't use map directly with this output, so we process it manually
    processed_data = []
    tokenized_results = tokenize_function(dataset)
    for i in range(len(dataset)):
        processed_data.append({
            "input_ids": tokenized_results["input_ids"][i],
            "labels": tokenized_results["labels"][i],
            "attention_mask": tokenized_results["attention_mask"][i],
            # For evaluation
            "question": dataset[i]['prompt'],
            "ground_truth": dataset[i]['full_response'],
        })

    # Split data: 80% train, 20% test
    train_size = int(0.8 * len(processed_data))
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]
    
    return train_data, test_data


def collate_fn(batch, tokenizer):
    """Custom collate function to handle padding."""
    # Pad to the max length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item['labels'] for item in batch],
        batch_first=True,
        padding_value=-100 # Important: pad labels with -100
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0 # Pad attention mask with 0
    )
    
    # Keep other data for evaluation
    questions = [item['question'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'questions': questions,
        'ground_truths': ground_truths
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, 
    tokenizer: PreTrainedTokenizer, 
    test_dataloader: DataLoader, 
    is_prompt_tuned: bool,
    log_callback: Optional[callable] = None
) -> float:
    """Evaluates model accuracy on the test set."""
    model.eval()
    correct = 0
    total = 0
    
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        questions = batch['questions']
        ground_truths = batch['ground_truths']
        
        # Tokenize only the question part for generation
        model_inputs = tokenizer(questions, return_tensors="pt", padding=True).to(TrainConfig.DEVICE)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if is_prompt_tuned:
                # Use the custom generate method for the wrapped model
                generated_ids = model.generate(**model_inputs, max_new_tokens=TrainConfig.MAX_NEW_TOKENS)
            else:
                # Use the base model's generate
                generated_ids = model.generate(**model_inputs, max_new_tokens=TrainConfig.MAX_NEW_TOKENS)
        
        # Decode and compare
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i in range(len(decoded_preds)):
            # Extract only the generated part
            #pred_text = decoded_preds[i][len(questions[i]):]
            pred_text = decoded_preds[i]
            
            pred_answer = extract_answer(pred_text)
            true_answer = extract_answer(ground_truths[i])
            
            if pred_answer is not None and true_answer is not None and pred_answer == true_answer:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    if log_callback:
        log_callback(f"Evaluation finished. Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy

# --- Main Training Function ---
def train(
    config: TrainConfig, 
    log_callback: callable, 
    stop_event: Any,
    continue_from_checkpoint: Optional[str] = None
):
    """
    Main training loop for prompt tuning.
    
    Args:
        config (TrainConfig): The configuration object.
        log_callback (callable): A function to send log messages to the UI.
        stop_event (threading.Event or similar): An event to signal training should stop.
        continue_from_checkpoint (str, optional): Path to a checkpoint to resume training.
    """
    try:
        # --- Experiment Setup ---
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"Qwen2.5-1.5B_pt{config.NUM_PROMPT_TOKENS}_lr{config.LEARNING_RATE}_e{config.NUM_EPOCHS}_{timestamp}"
        exp_dir = os.path.join(config.BASE_DIR, exp_name)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        log_dir = os.path.join(exp_dir, "logs")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        log_callback(f"Starting experiment: {exp_name}")
        log_callback(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")

        # --- Model and Tokenizer ---
        log_callback("Loading base model and tokenizer...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            torch_dtype=config.DTYPE,
            device_map=config.DEVICE
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Wrapped Model ---
        model = PromptTunedModel(base_model, config.NUM_PROMPT_TOKENS).to(config.DEVICE)
        
        start_epoch = 0
        if continue_from_checkpoint:
            log_callback(f"Resuming from checkpoint: {continue_from_checkpoint}")
            try:
                # The checkpoint only contains the soft prompt state dict
                ckpt_state_dict = torch.load(continue_from_checkpoint, map_location=config.DEVICE)
                model.load_state_dict(ckpt_state_dict, strict=True)
                
                # Try to infer epoch from checkpoint name, e.g., "epoch_1.pt"
                match = re.search(r'epoch_(\d+)', continue_from_checkpoint)
                if match:
                    start_epoch = int(match.group(1)) + 1
                    log_callback(f"Resuming from end of epoch {start_epoch - 1}")
            except Exception as e:
                log_callback(f"Error loading checkpoint: {e}. Starting from scratch.")
                start_epoch = 0

        log_callback(f"Trainable parameters: {[name for name, param in model.named_parameters() if param.requires_grad]}")

        # --- Data ---
        log_callback("Preparing dataset...")
        train_data, test_data = prepare_data(tokenizer, config.NUM_TEST_SAMPLES)
        
        # Use a lambda to pass the tokenizer to the collate function
        collate_with_tokenizer = lambda batch: collate_fn(batch, tokenizer)
        
        train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_with_tokenizer)
        # For evaluation, we process one by one, so batch size is 1
        test_dataloader = DataLoader(test_data, batch_size=1, collate_fn=collate_with_tokenizer)
        
        log_callback(f"Dataset prepared: {len(train_data)} train, {len(test_data)} test samples.")

        # --- Optimizer and Logger ---
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
        writer = SummaryWriter(log_dir)
        global_step = 0

        # --- Training Loop ---
        log_callback("Starting training...")
        model.train()
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            if stop_event.is_set():
                log_callback("Training stopped by user.")
                break
            
            log_callback(f"--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                if stop_event.is_set():
                    break
                    
                optimizer.zero_grad()
                
                # Move batch to device
                input_ids = batch['input_ids'].to(config.DEVICE)
                attention_mask = batch['attention_mask'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                # Logging
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                writer.add_scalar("Loss/train", loss.item(), global_step)
                if global_step % 10 == 0:
                    log_callback(f"Step {global_step}, Loss: {loss.item():.4f}")
                global_step += 1

            # --- End of Epoch Evaluation and Checkpointing ---
            if not stop_event.is_set():
                log_callback(f"Evaluating at end of epoch {epoch+1}...")
                
                # Evaluate on a subset of training data for quick feedback
                train_acc = evaluate(model, tokenizer, DataLoader(train_data[:20], batch_size=1, collate_fn=collate_with_tokenizer), is_prompt_tuned=True)
                log_callback(f"Train Accuracy (20 samples): {train_acc:.4f}")
                writer.add_scalar("Accuracy/train", train_acc, epoch)

                # Evaluate on full test set
                test_acc = evaluate(model, tokenizer, test_dataloader, is_prompt_tuned=True)
                log_callback(f"Test Accuracy: {test_acc:.4f}")
                writer.add_scalar("Accuracy/test", test_acc, epoch)
                
                # Save checkpoint
                ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
                # We only need to save the soft prompt parameters
                torch.save(model.state_dict(), ckpt_path)
                log_callback(f"Checkpoint saved to {ckpt_path}")
        
        writer.close()
        log_callback("Training finished.")

    except Exception as e:
        log_callback(f"An error occurred during training: {e}")
        import traceback
        log_callback(traceback.format_exc())

# --- Unit Test ---
if __name__ == "__main__":
    
    def dummy_log(message):
        print(f"[LOG] {message}")
        
    class DummyStopEvent:
        def is_set(self): return False

    print("--- Running train.py Unit Test ---")
    
    config = TrainConfig()
    config.NUM_TEST_SAMPLES = 10 # Use a small number of samples for the test

    # 1. Load base model and tokenizer
    print("\n[Step 1] Loading base model and tokenizer...")
    base_model_test = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=config.DTYPE,
        device_map=config.DEVICE
    )
    tokenizer_test = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer_test.pad_token is None:
        tokenizer_test.pad_token = tokenizer_test.eos_token
    print("Base model and tokenizer loaded.")
    
    # 2. Get a sample question
    print("\n[Step 2] Preparing a sample from GSM8K...")
    _, test_data_sample = prepare_data(tokenizer_test, 1)
    sample_item = test_data_sample[0]
    sample_question = sample_item['question']
    sample_ground_truth = sample_item['ground_truth']
    
    print(f"Sample Question:\n---\n{sample_question}\n---")
    print(f"Sample Ground Truth:\n---\n{sample_ground_truth}\n---")

    # 3. Test 1: Inference with base model (no prompt tuning) using Flash Attention
    print("\n[Step 3] Running reference inference with base model + Flash Attention...")
    model_inputs = tokenizer_test([sample_question], return_tensors="pt").to(config.DEVICE)
    
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        print("Flash Attention 2 enabled for generation.")
        generated_ids_base = base_model_test.generate(**model_inputs, max_new_tokens=512)
        
    response_base = tokenizer_test.batch_decode(generated_ids_base, skip_special_tokens=True)[0]
    #response_base_only_new = response_base[len(sample_question):]
    response_base_only_new = response_base
    
    print("\n--- LLM Response (Base Model) ---")
    print(response_base_only_new)
    print("---------------------------------")
    
    # 4. Test 2: Inference with a dummy prompt-tuned model (all-zero soft prompt)
    print("\n[Step 4] Running inference with dummy (all-zero) soft prompt...")
    
    # Create a PromptTunedModel instance
    pt_model_test = PromptTunedModel(base_model_test, config.NUM_PROMPT_TOKENS).to(config.DEVICE)
    
    # Create a dummy checkpoint (soft prompt tensor with all zeros)
    dummy_soft_prompt = torch.zeros_like(pt_model_test.soft_prompt)
    pt_model_test.load_state_dict({"soft_prompt": dummy_soft_prompt}, strict=False)
    pt_model_test.eval()
    
    print("Dummy prompt-tuned model created with zeroed soft prompt.")
    
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        print("Flash Attention 2 enabled for generation.")
        # The generate method of PromptTunedModel handles inputs_embeds
        generated_ids_pt = pt_model_test.generate(**model_inputs, max_new_tokens=512)
        
    response_pt = tokenizer_test.batch_decode(generated_ids_pt, skip_special_tokens=True)[0]
    #response_pt_only_new = response_pt[len(sample_question):]
    response_pt_only_new = response_pt

    print("\n--- LLM Response (Dummy Checkpoint) ---")
    print(response_pt_only_new)
    print("---------------------------------------")
    
    print("\nUnit test finished. If both tests ran without errors and produced output,")
    print("the core logic for both base and prompt-tuned inference is working.")
    print("The dummy checkpoint response is expected to be different (and likely nonsensical).")
