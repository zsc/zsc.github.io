# app.py
import os
import glob
import logging
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from multiprocessing import Process, Queue, Event
import argparse

# Import the training and evaluation functions from your script
from train import train, evaluate

# --- Flask App Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Global State Management ---
# Using a dictionary to hold state variables
# This makes it easier to manage and reset state
state = {
    "process": None,
    "log_queue": None,
    "stop_event": None
}

# --- Utility Functions ---
def is_process_running():
    """Checks if a process is currently active."""
    return state["process"] is not None and state["process"].is_alive()

def terminate_process():
    """Stops the currently running process."""
    if is_process_running():
        logging.info("Terminating existing process...")
        state["stop_event"].set()
        try:
            # Wait for a bit for graceful shutdown
            state["process"].join(timeout=10)
        except Exception as e:
            logging.error(f"Error joining process: {e}")
        
        if state["process"].is_alive():
            logging.warning("Process did not terminate gracefully. Forcing termination.")
            state["process"].terminate()
            state["process"].join() # Ensure it's fully gone

    state["process"] = None
    state["log_queue"] = None
    state["stop_event"] = None
    logging.info("Process terminated and state cleared.")

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main UI page."""
    return render_template('index.html')

@app.route('/get_checkpoints')
def get_checkpoints():
    """Scans for and returns a list of available .pt checkpoint files."""
    checkpoints_dir = './checkpoints'
    if not os.path.isdir(checkpoints_dir):
        return jsonify([])
    # Use glob to recursively find all .pt files
    checkpoints = glob.glob(os.path.join(checkpoints_dir, '**', '*.pt'), recursive=True)
    # Normalize path separators for web
    checkpoints = [p.replace('\\', '/') for p in checkpoints]
    return jsonify(sorted(checkpoints, reverse=True))

@app.route('/start_train', methods=['POST'])
def start_train():
    """Starts a new training process."""
    if is_process_running():
        return jsonify({"status": "error", "message": "A process is already running."}), 400

    try:
        # Collect hyperparameters from the form
        args = argparse.Namespace(
            epochs=int(request.form.get('epochs', 3)),
            batch_size=int(request.form.get('batch_size', 1)),
            learning_rate=float(request.form.get('learning_rate', 1e-3)),
            weight_decay=0.01,
            num_prompt_tokens=int(request.form.get('num_prompt_tokens', 20)),
            max_length=512,
            checkpoints_dir='./checkpoints',
            save_steps=7000,
            num_train_samples=7000,
            num_test_samples=100,
            checkpoint_path=request.form.get('continue_from_checkpoint') or None,
            max_new_tokens=256
        )

        # Setup multiprocessing components
        state["log_queue"] = Queue()
        state["stop_event"] = Event()
        
        # Create and start the process
        state["process"] = Process(target=train, args=(args, state["stop_event"], state["log_queue"]))
        state["process"].start()

        logging.info(f"Started training process with PID: {state['process'].pid}")
        return jsonify({"status": "success", "message": "Training started."})

    except Exception as e:
        logging.error(f"Failed to start training: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/start_test', methods=['POST'])
def start_test():
    """Starts a new evaluation process."""
    if is_process_running():
        return jsonify({"status": "error", "message": "A process is already running."}), 400

    checkpoint_path = request.form.get('test_checkpoint_path')
    if not checkpoint_path:
        return jsonify({"status": "error", "message": "No checkpoint selected for testing."}), 400

    try:
        args = argparse.Namespace(
            checkpoint_path=checkpoint_path,
            num_test_samples=100,
            max_new_tokens=256
        )
        
        state["log_queue"] = Queue()
        state["stop_event"] = Event() # Not used by evaluate, but good practice

        state["process"] = Process(target=evaluate, args=(args, None, state["log_queue"]))
        state["process"].start()

        logging.info(f"Started evaluation process with PID: {state['process'].pid}")
        return jsonify({"status": "success", "message": "Evaluation started."})

    except Exception as e:
        logging.error(f"Failed to start evaluation: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stop_process', methods=['POST'])
def stop_process():
    """Stops the currently running train or test process."""
    if not is_process_running():
        return jsonify({"status": "error", "message": "No process is currently running."}), 400
    
    terminate_process()
    return jsonify({"status": "success", "message": "Process stopping."})

@app.route('/log_stream')
def log_stream():
    """Streams logs from the queue to the client using Server-Sent Events."""
    def generate():
        if not state["log_queue"]:
            yield "data: No active process to log.\n\n"
            return
        
        while is_process_running():
            try:
                # Non-blocking get from queue
                log_line = state["log_queue"].get(timeout=1)
                yield f"data: {log_line}\n\n"
                if "---PROCESS-COMPLETE---" in log_line:
                    break
            except Exception:
                # Timeout, just check if process is still alive
                continue

        # After the loop, clear the state
        terminate_process()
        yield "data: ---STREAM-END---\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    # Ensure checkpoint directory exists
    os.makedirs('checkpoints', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
