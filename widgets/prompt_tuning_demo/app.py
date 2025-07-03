# app.py

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import threading
import time
import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.nn.attention import SDPBackend, sdpa_kernel
from datasets import load_dataset
import torch
import traceback

# Import SLOT logic from train.py
from train import get_model_and_tokenizer, format_prompt, extract_answer, SLOTOptimizer, MODEL_NAME

# --- Flask App and SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# --- Global Variables ---
# To hold the background evaluation thread
eval_thread = None
# A flag to signal the thread to stop
stop_flag = threading.Event()

# To hold the loaded model and prevent reloading
MODEL_GLOBAL = None
TOKENIZER_GLOBAL = None
SLOT_OPTIMIZER_GLOBAL = None

# Base directory for storing experiment results
EXPERIMENTS_DIR = "experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

def load_globals_if_needed(log_fn=print):
    """Loads model and tokenizer into global variables if they aren't already."""
    global MODEL_GLOBAL, TOKENIZER_GLOBAL
    if MODEL_GLOBAL is None or TOKENIZER_GLOBAL is None:
        log_fn("Loading model and tokenizer for the first time... This may take a moment.")
        MODEL_GLOBAL, TOKENIZER_GLOBAL = get_model_and_tokenizer(MODEL_NAME)
        log_fn("Model and tokenizer are loaded and ready.")
    else:
        log_fn("Model and tokenizer already in memory.")


# --- Background Evaluation Task ---
def run_batch_evaluation(config):
    """The main function that runs in a background thread."""
    try:
        # Unpack config
        T = int(config['T'])
        lr = float(config['lr'])
        num_samples = int(config['num_samples'])
        resume_dir = config.get('resume_dir')

        start_index = 0
        results = []
        
        if resume_dir:
            exp_dir = os.path.join(EXPERIMENTS_DIR, resume_dir)
            log_path = os.path.join(exp_dir, "eval_log.txt")
            progress_path = os.path.join(exp_dir, "progress.json")
            socketio.emit('log_message', {'data': f"Resuming evaluation from: {resume_dir}"})
            if os.path.exists(progress_path):
                with open(progress_path, 'r') as f:
                    progress = json.load(f)
                    start_index = progress['last_completed_index'] + 1
                    results = progress['results']
                socketio.emit('log_message', {'data': f"Loaded progress. Resuming from sample {start_index + 1}."})
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            exp_name = f"slot_eval_T{T}_lr{lr}_{timestamp}"
            exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            log_path = os.path.join(exp_dir, "eval_log.txt")
            socketio.emit('log_message', {'data': f"Starting new evaluation. Results will be saved in: {exp_dir}"})

        # Helper to log to both file and socket
        def log(message):
            socketio.emit('log_message', {'data': message})
            with open(log_path, 'a') as f:
                f.write(message + '\n')

        load_globals_if_needed(log)
        
        # Setup TensorBoard
        writer = SummaryWriter(log_dir=exp_dir)

        # Load dataset
        dataset = load_dataset("gsm8k", "main", split=f"test[:{num_samples}]")
        
        slot_optimizer = SLOTOptimizer(model=MODEL_GLOBAL, tokenizer=TOKENIZER_GLOBAL, T=T, lr=lr)
        
        total_samples = len(dataset)
        slot_correct = sum(1 for r in results if r['slot_correct'])
        
        for i in range(start_index, total_samples):
            if stop_flag.is_set():
                log("Evaluation stopped by user.")
                break
            
            sample_start_time = time.time()
            sample = dataset[i]
            question = sample['question']
            ground_truth_answer = extract_answer(sample['answer'])
            
            log(f"\n--- Processing Sample {i+1}/{total_samples} ---")
            
            prompt_text = format_prompt(question)
            
            # SLOT Generation
            log("Optimizing and generating with SLOT...")
            optimized_response, prompt_losses = slot_optimizer.optimize_and_generate(
                prompt_text=prompt_text, max_new_tokens=512
            )
            slot_answer = extract_answer(optimized_response)
            is_correct_slot = slot_answer == ground_truth_answer
            if is_correct_slot:
                slot_correct += 1

            slot_accuracy = (slot_correct / (i + 1)) * 100
            
            # Logging and progress update
            log(f"Ground Truth: {ground_truth_answer}, SLOT Answer: {slot_answer}, Correct: {is_correct_slot}")
            log(f"Sample took {time.time() - sample_start_time:.2f}s. Current Accuracy: {slot_accuracy:.2f}%")

            # TensorBoard
            writer.add_scalar('Accuracy/SLOT', slot_accuracy, i)
            try:
                writer.add_scalar('Loss/Prompt_Optimization_Final', prompt_losses[-1], i)
            except Exception as e:
                print(e)
            
            # Save results and progress
            current_result = {
                'index': i,
                'question': question,
                'ground_truth': ground_truth_answer,
                'slot_answer': slot_answer,
                'slot_correct': is_correct_slot
            }
            results.append(current_result)
            
            progress_path = os.path.join(exp_dir, "progress.json")
            with open(progress_path, 'w') as f:
                json.dump({'last_completed_index': i, 'results': results}, f, indent=2)

            socketio.emit('status_update', {
                'progress': f"{i+1}/{total_samples}",
                'accuracy': f"{slot_accuracy:.2f}%"
            })

        log("\n--- Evaluation Finished ---")
        final_accuracy = (slot_correct / (i + 1 if i >= start_index else 1)) * 100
        log(f"Final SLOT Accuracy: {slot_correct}/{i+1} = {final_accuracy:.2f}%")
        writer.close()

    except Exception as e:
        tb_str = traceback.format_exc()
        socketio.emit('log_message', {'data': f"An error occurred: {str(e)}"})
        socketio.emit('log_message', {'data': tb_str})
    finally:
        socketio.emit('eval_finished')


def run_single_test(config):
    """Runs a single test on a user-provided prompt."""
    try:
        prompt = config['prompt']
        T = int(config['T'])
        lr = float(config['lr'])

        def log(message):
            socketio.emit('test_log', {'data': message})

        load_globals_if_needed(log)
        slot_optimizer = SLOTOptimizer(model=MODEL_GLOBAL, tokenizer=TOKENIZER_GLOBAL, T=T, lr=lr)
        
        # 1. Base model generation
        log("Generating with original model (Flash Attention)...")
        start_time = time.time()
        chat_prompt = TOKENIZER_GLOBAL.apply_chat_template([{'role':'user', 'content':prompt}], tokenize=False, add_generation_prompt=True)
        model_inputs = TOKENIZER_GLOBAL([chat_prompt], return_tensors="pt").to(MODEL_GLOBAL.device)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            generated_ids = MODEL_GLOBAL.generate(
                model_inputs.input_ids, max_new_tokens=512, do_sample=False
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        base_response = TOKENIZER_GLOBAL.decode(output_ids, skip_special_tokens=True)
        log(f"Original model generation took {time.time() - start_time:.2f}s.")

        # 2. SLOT optimized generation
        log("Optimizing and generating with SLOT...")
        start_time = time.time()
        slot_response, _ = slot_optimizer.optimize_and_generate(chat_prompt, 512)
        log(f"SLOT generation took {time.time() - start_time:.2f}s.")

        socketio.emit('test_result', {'base': base_response, 'slot': slot_response})

    except Exception as e:
        tb_str = traceback.format_exc()
        socketio.emit('test_log', {'data': f"An error occurred: {str(e)}"})
        socketio.emit('test_log', {'data': tb_str})
    finally:
        socketio.emit('test_finished')


# --- Flask Routes and SocketIO Events ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_experiments')
def get_experiments():
    dirs = [d for d in os.listdir(EXPERIMENTS_DIR) if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))]
    return jsonify(sorted(dirs, reverse=True))

@socketio.on('start_eval')
def handle_start_eval(json_data):
    global eval_thread
    if eval_thread and eval_thread.is_alive():
        socketio.emit('log_message', {'data': "An evaluation is already in progress."})
        return

    stop_flag.clear()
    eval_thread = socketio.start_background_task(run_batch_evaluation, json_data)
    socketio.emit('eval_started')

@socketio.on('stop_eval')
def handle_stop_eval():
    if eval_thread and eval_thread.is_alive():
        stop_flag.set()
        socketio.emit('log_message', {'data': "Stop signal sent. Finishing current sample..."})
    else:
        socketio.emit('log_message', {'data': "No evaluation is running."})

@socketio.on('run_test')
def handle_run_test(json_data):
    socketio.start_background_task(run_single_test, json_data)

if __name__ == '__main__':
    print("Starting Flask app. Open http://127.0.0.1:5000 in your browser.")
    socketio.run(app, debug=False)
