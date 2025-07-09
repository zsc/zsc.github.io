import os
import uuid
import traceback  # Import the traceback module
from pathlib import Path
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename

# Import the Rife class from your provided rife.py
from rife import Rife

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
MODEL_DIR = 'train_log'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256 MB upload limit
app.secret_key = 'super-secret-key-for-flashing' # Necessary for flash messages

# Create necessary directories if they don't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# --- Model Initialization ---
# Initialize the RIFE model once when the application starts.
# This is crucial for performance as model loading can be slow.
# The `rife.py` script automatically handles CUDA/MPS/CPU device selection.
# As requested, we assume a 4090 and use float32 (use_fp16=False).
print("Initializing RIFE model... This may take a moment.")
try:
    rife_interpolator = Rife(model_dir=MODEL_DIR, use_fp16=False)
    print("RIFE model loaded successfully.")
except Exception as e:
    print(f"FATAL: Failed to initialize RIFE model: {e}")
    # Print the full stack trace to the console for detailed debugging
    traceback.print_exc()
    rife_interpolator = None

# --- Helper Function ---
def allowed_file(filename):
    """Checks if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not rife_interpolator:
            flash('Error: RIFE model is not available. Please check server logs.', 'error')
            return render_template('index.html')

        # --- 1. Handle File Upload ---
        if 'video_file' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(request.url)
        file = request.files['video_file']
        if file.filename == '':
            flash('No video file selected.', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Sanitize filename and create a unique path
            original_filename = secure_filename(file.filename)
            unique_id = uuid.uuid4().hex[:8]
            input_filename = f"{Path(original_filename).stem}_{unique_id}{Path(original_filename).suffix}"
            input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            file.save(input_filepath)

            # --- 2. Get Hyperparameters from Form ---
            try:
                multiplier = request.form.get('multiplier', 2, type=int)
                scale = request.form.get('scale', 1.0, type=float)
            except (ValueError, TypeError):
                flash('Invalid hyperparameter value.', 'error')
                return redirect(request.url)

            # --- 3. Run RIFE Processing ---
            try:
                print(f"Processing '{input_filepath}' with multiplier={multiplier}, scale={scale}")
                output_video_path = rife_interpolator.process(
                    input_video_path=input_filepath,
                    multiplier=multiplier,
                    scale=scale,
                    output_dir=app.config['OUTPUT_FOLDER']
                )
                output_filename = os.path.basename(output_video_path)
                print(f"Processing complete. Output file: {output_filename}")

                # --- 4. Show Result ---
                # Pass the output filename to the template to generate a download link
                return render_template('index.html', processed_file=output_filename)

            except Exception as e:
                # Catch errors from the RIFE process (e.g., ffmpeg errors)
                print(f"Error during RIFE processing: {e}")
                # Print the full stack trace to the console for detailed debugging
                traceback.print_exc()
                flash(f'An error occurred during processing: {e}', 'error')
                return redirect(request.url)
        else:
            flash(f'Invalid file type. Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
            return redirect(request.url)

    # Handle GET request
    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    """Serves the processed video for download."""
    # Basic security: prevent directory traversal attacks
    safe_path = Path(app.config['OUTPUT_FOLDER']) / Path(filename).name
    if not safe_path.is_file():
        flash('File not found.', 'error')
        return redirect(url_for('index'))
        
    return send_from_directory(
        app.config['OUTPUT_FOLDER'],
        filename,
        as_attachment=True
    )

if __name__ == '__main__':
    # Running with debug=False is recommended for performance,
    # as it prevents the app (and the model) from reloading on code changes.
    # Use host='0.0.0.0' to make it accessible on your local network.
    app.run(host='0.0.0.0', port=5000, debug=False)
