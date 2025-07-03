# train.py

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import time
import os
from tqdm import tqdm

# --- Constants and Hyperparameters ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
NUM_SAMPLES_FOR_UNIT_TEST = 3 # Number of samples for a quick test run
MAX_NEW_TOKENS = 512 # Max tokens for the generated answer

# SLOT Hyperparameters
T_OPTIMIZATION_STEPS = 10
LEARNING_RATE = 1e-2 # A higher learning rate is often used for test-time tuning

# --- Helper Functions ---

def get_model_and_tokenizer(model_name: str):
    """Loads the pre-trained model and tokenizer."""
    print(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded.")
    return model, tokenizer

def format_prompt(question: str) -> str:
    """Formats the question into the Qwen chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves math problems step by step and then write the final answer in the format \\boxed{<number>}."},
        {"role": "user", "content": f"Question: {question}"}
    ]
    # The tokenizer will apply the template and add the generation prompt.
    text = AutoTokenizer.from_pretrained(MODEL_NAME).apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def extract_answer(text: str) -> str:
    """Extracts the final numerical answer from the model's generation."""
    # The pattern looks for the number inside \boxed{}
    match = re.search(r"\\boxed{([,\d\.\-]+)}", text)
    if match:
        # Remove commas and return the number
        return match.group(1).replace(",", "")
    # Fallback if the format is not found
    # Look for the last number in the string
    matches = re.findall(r"[\d\.\-]+", text)
    return matches[-1] if matches else ""


# --- SLOT Optimizer Class ---

class SLOTOptimizer:
    """
    Implements the SLOT algorithm for a single sample.
    """
    def __init__(self, model, tokenizer, T: int, lr: float):
        """
        Initializes the optimizer.
        Args:
            model: The pre-trained Hugging Face model.
            tokenizer: The corresponding tokenizer.
            T: Number of optimization steps for delta.
            lr: Learning rate for the delta optimizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.T = T
        self.lr = lr
        self.device = model.device
        self.hidden_size = model.config.hidden_size
        self.vocab_size = model.config.vocab_size

        # For convenience, separate the model body (transformer blocks) from the language model head
        self.model_body = model.model
        self.lm_head = model.lm_head

    def _optimize_delta(self, input_ids: torch.Tensor):
        """
        Performs Stage 1: Prompt Stage - Optimizes the sample-specific delta parameter.
        """
        # 1. Initialize delta and optimizer
        # Delta is a single vector of size (1, 1, d) that will be broadcasted.
        delta = torch.zeros(
            1, 1, self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True
        )
        optimizer = torch.optim.AdamW([delta], lr=self.lr)

        # 2. Pre-compute and cache hidden states H for the prompt
        with torch.no_grad():
            # H has shape (batch_size, seq_len, hidden_size)
            H = self.model_body(input_ids=input_ids).last_hidden_state

        # 3. Optimize delta for T steps
        for _ in range(self.T):
            optimizer.zero_grad()

            # Add delta to the cached hidden states
            H_prime = H + delta
            
            # Get logits using the modified hidden states
            # W_LM^T is W_LM.T in PyTorch
            logits = self.lm_head(H_prime)

            # Calculate Cross-Entropy loss on the prompt itself
            # We predict the next token, so shift logits and labels
            # Logits shape: (1, n, vocab_size), Labels shape: (1, n)
            pred_logits = logits[:, :-1, :] # Shape: (1, n-1, vocab_size)
            target_ids = input_ids[:, 1:]   # Shape: (1, n-1)

            loss = F.cross_entropy(
                pred_logits.reshape(-1, self.vocab_size),
                target_ids.reshape(-1)
            )
            
            # Backpropagate and update delta
            loss.backward()
            optimizer.step()

        # 4. Return the optimized delta, detached from the computation graph
        return delta.detach(), loss.detach()

    def _generate_with_delta(self, input_ids: torch.Tensor, delta_opt: torch.Tensor, max_new_tokens: int) -> str:
        """
        Performs Stage 2: Generation Stage - Generates a response using the optimized delta.
        This uses a custom autoregressive generation loop to inject delta at each step.
        """
        generated_ids = []
        current_ids = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Get model outputs for the current sequence
                if past_key_values:
                    # If we have a KV cache, only process the last token
                    model_input = current_ids[:, -1:]
                else:
                    # First step, process the entire prompt
                    model_input = current_ids
                
                outputs = self.model_body(
                    input_ids=model_input,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                
                hidden_states = outputs.last_hidden_state
                past_key_values = outputs.past_key_values

            # --- FIX STARTS HERE ---
            # The original code used all hidden_states, causing a shape mismatch on the first pass.
            # We must select only the *last token's* hidden state to predict the next token.
            # This slices the tensor from shape (batch, seq_len, hidden_size)
            # to (batch, 1, hidden_size), which is what we need.
            last_token_hidden_state = hidden_states[:, -1:, :]
            # --- FIX ENDS HERE ---

            # Apply the optimized delta to the last hidden state
            h_prime_last = last_token_hidden_state + delta_opt

            # Calculate logits for the next token
            next_token_logits = self.lm_head(h_prime_last)

            # Greedy decoding: select the token with the highest probability
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Check for end-of-sequence token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Append the new token and update the current sequence for the next iteration
            generated_ids.append(next_token.item())
            current_ids = next_token

        # Decode the generated token IDs into text
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def optimize_and_generate(self, prompt_text: str, max_new_tokens: int) -> str:
        """
        The main method that runs the full SLOT pipeline for a given prompt.
        """
        # Tokenize the input prompt
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        input_ids = model_inputs.input_ids

        # --- Stage 1: Optimize delta ---
        print("  Optimizing delta...")
        start_time = time.time()
        delta_opt, loss = self._optimize_delta(input_ids)
        print(f"  Delta optimization took {time.time() - start_time:.2f} seconds.")

        # --- Stage 2: Generate with optimized delta ---
        print("  Generating with optimized delta...")
        start_time = time.time()
        generated_text = self._generate_with_delta(input_ids, delta_opt, max_new_tokens)
        print(f"  Generation with delta took {time.time() - start_time:.2f} seconds.")

        return generated_text, loss


# --- Unit Test ---

def run_unit_test():
    """
    Runs a unit test on a few samples from the GSM8K dataset.
    It compares the original model's generation with the SLOT-optimized generation.
    """
    print("--- Running Unit Test for SLOT Algorithm ---")

    # 1. Load model, tokenizer, and dataset
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    # 2. Instantiate SLOT Optimizer
    slot_optimizer = SLOTOptimizer(
        model=model,
        tokenizer=tokenizer,
        T=T_OPTIMIZATION_STEPS,
        lr=LEARNING_RATE
    )

    original_correct_count = 0
    slot_correct_count = 0

    # 3. Loop through a few samples for demonstration
    for i in range(NUM_SAMPLES_FOR_UNIT_TEST):
        sample = dataset[i]
        question = sample['question']
        ground_truth_full = sample['answer']
        ground_truth_answer = extract_answer(ground_truth_full)

        print("\n" + "="*80)
        print(f"--- Sample {i+1}/{NUM_SAMPLES_FOR_UNIT_TEST} ---")
        print(f"Question: {question}")
        print(f"Ground Truth Answer: {ground_truth_answer}")
        print("-" * 30)

        prompt_text = format_prompt(question)
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        # --- A. Reference Generation (delta=0) using Flash Attention 2 ---
        # This is the "idiot-proof" reference answer from the original model.
        print("\n>>> Generating with Original Model (δ=0, Flash Attention 2) <<<")
        start_time = time.time()
        # Use sdpa_kernel context manager to enable Flash Attention
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False  # Use greedy decoding for consistency
            )
        print(f"  Reference generation took {time.time() - start_time:.2f} seconds.")
        
        # model.generate's output includes the input prompt, so we slice it off.
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        reference_response = tokenizer.decode(output_ids, skip_special_tokens=True)
        reference_answer_val = extract_answer(reference_response)
        is_correct_original = reference_answer_val == ground_truth_answer
        if is_correct_original:
            original_correct_count += 1
        
        print("\n--- LLM Response (Original) ---")
        print(reference_response)
        print("--------------------------------")
        print(f"Extracted Answer (Original): {reference_answer_val}")
        print(f"Correct: {is_correct_original}")

        # --- B. SLOT Optimized Generation ---
        print("\n>>> Generating with SLOT Optimized Model (δ_opt) <<<")
        optimized_response, _ = slot_optimizer.optimize_and_generate(
            prompt_text=prompt_text,
            max_new_tokens=MAX_NEW_TOKENS
        )
        optimized_answer_val = extract_answer(optimized_response)
        is_correct_slot = optimized_answer_val == ground_truth_answer
        if is_correct_slot:
            slot_correct_count += 1

        print("\n--- LLM Response (SLOT Optimized) ---")
        print(optimized_response)
        print("---------------------------------------")
        print(f"Extracted Answer (SLOT): {optimized_answer_val}")
        print(f"Correct: {is_correct_slot}")
        print("="*80)


    # --- Final Accuracy Summary ---
    print("\n" + "#"*80)
    print("--- Final Accuracy Summary ---")
    print(f"Tested on {NUM_SAMPLES_FOR_UNIT_TEST} samples from {DATASET_NAME} '{DATASET_SPLIT}' split.")

    original_accuracy = (original_correct_count / NUM_SAMPLES_FOR_UNIT_TEST) * 100
    print(f"\nOriginal Model Accuracy: {original_correct_count}/{NUM_SAMPLES_FOR_UNIT_TEST} = {original_accuracy:.2f}%")

    slot_accuracy = (slot_correct_count / NUM_SAMPLES_FOR_UNIT_TEST) * 100
    print(f"SLOT Optimized Model Accuracy: {slot_correct_count}/{NUM_SAMPLES_FOR_UNIT_TEST} = {slot_accuracy:.2f}%")
    print("#"*80)

if __name__ == "__main__":
    run_unit_test()
