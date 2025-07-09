import os
import sys
import torch
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from tqdm import tqdm
import torch.nn.functional as F
import traceback

# Add the parent directory to the system path to allow for relative imports.
sys.path.append(str(Path(__file__).resolve().parent))

# Import model definitions and the SSIM utility.
try:
    from train_log.RIFE_HDv3 import Model as RIFE_Model
    from model.pytorch_msssim import ssim_matlab
    from model import warplayer
except ImportError:
    print("Error: Could not import RIFE model components or pytorch_msssim.")
    print("Please ensure the following files/directories exist relative to this script:")
    print("- train_log/RIFE_HDv3.py")
    print("- train_log/IFNet_HDv3.py")
    print("- model/warplayer.py")
    print("- model/pytorch_msssim.py")
    sys.exit(1)

# --- Device and Data Type Configuration ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float32
    print("Device: CUDA")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float16  # MPS performs better with float16
    print("Device: Apple MPS")
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    print("Device: CPU")

class Rife:
    """
    A wrapper class for the RIFE frame interpolation model, optimized for video processing.
    Incorporates advanced techniques like scene-change detection and robust audio transfer.
    """
    def __init__(self, model_dir: str = 'train_log/', use_fp16: bool = False):
        """
        Initializes the RIFE model.

        Args:
            model_dir (str): The directory containing the 'flownet.pkl' model checkpoint.
            use_fp16 (bool): Whether to use half-precision floating-point format (float16).
        """
        self.model_dir = model_dir
        self.device = DEVICE
        self.dtype = DTYPE if use_fp16 and self.device != torch.device('cpu') else torch.float32

        print(f"Initializing RIFE model on device '{self.device}' with dtype '{self.dtype}'...")

        self.model = RIFE_Model()
        self.model.load_model(self.model_dir)
        self.model.eval()

        self.model.flownet.to(self.device, dtype=self.dtype)

        print("RIFE model initialized successfully.")

    def _pad_image(self, img_tensor: torch.Tensor):
        b, c, h, w = img_tensor.shape
        # RIFE models require dimensions to be divisible by a certain number (e.g., 64).
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h != 0 or pad_w != 0:
            return F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate'), (w, h)
        return img_tensor, (w, h)

    def _preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        img_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        tensor = torch.clamp(tensor, 0, 1)
        return Image.fromarray((tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8))

    def _calculate_ssim(self, img0: torch.Tensor, img1: torch.Tensor) -> float:
        """Calculates SSIM between two tensors on the GPU for efficiency."""
        # Use smaller, resized tensors for faster SSIM calculation.
        img0_small = F.interpolate(img0, (64, 64), mode='bilinear', align_corners=False)
        img1_small = F.interpolate(img1, (64, 64), mode='bilinear', align_corners=False)
        # ssim_matlab expects (B, C, H, W) and handles RGB channels correctly.
        return ssim_matlab(img0_small[:, :3], img1_small[:, :3]).item()

    def _generate_interpolated_frames(self, img0: torch.Tensor, img1: torch.Tensor, multiplier: int, scale: float):
        """Generates intermediate frames using a direct iterative approach."""
        if multiplier == 1:
            return []
        
        with torch.no_grad():
            output_tensors = []
            for i in range(multiplier - 1):
                timestep = (i + 1) / multiplier
                # The model's inference function can generate a frame at any timestep.
                mid = self.model.inference(img0.float(), img1.float(), timestep, scale)
                output_tensors.append(mid.to(self.dtype))
        return output_tensors

    def _execute_ffmpeg_command(self, command):
        """Executes an FFmpeg command and handles potential errors."""
        try:
            process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            return process
        except FileNotFoundError:
            print("Error: ffmpeg command not found.")
            print("Please ensure ffmpeg is installed and accessible in your system's PATH.")
            raise
        except subprocess.CalledProcessError as e:
            print(f"Error during FFmpeg execution. Command: {' '.join(command)}")
            print(f"FFmpeg exited with code {e.returncode}")
            print(f"FFmpeg stderr:\n{e.stderr}")
            raise
            
    def _transfer_audio(self, source_path: str, no_audio_path: str, final_path: str):
        """
        Extracts audio from the source video and merges it with the interpolated video.
        Includes fallbacks for different audio codecs.
        """
        print("Attempting to transfer audio...")
        temp_dir = Path(final_path).parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_audio_file = temp_dir / "audio.mkv"

        # 1. Extract audio from source using a direct copy
        extract_cmd = ['ffmpeg', '-y', '-i', source_path, '-c:a', 'copy', '-vn', str(temp_audio_file)]
        self._execute_ffmpeg_command(extract_cmd)

        # 2. Merge interpolated video with the extracted audio
        merge_cmd = ['ffmpeg', '-y', '-i', no_audio_path, '-i', str(temp_audio_file), '-c', 'copy', final_path]
        try:
            self._execute_ffmpeg_command(merge_cmd)
            # Check if merge was successful by seeing if the file was created and is not empty
            if not (os.path.exists(final_path) and os.path.getsize(final_path) > 0):
                raise subprocess.CalledProcessError(1, merge_cmd, "FFmpeg created a zero-size file.")
            print("Lossless audio transfer successful.")
        
        except subprocess.CalledProcessError:
            print("Warning: Lossless audio transfer failed. Trying to transcode audio to AAC.")
            temp_audio_file_aac = temp_dir / "audio.m4a"
            # 3. If direct copy fails, transcode audio to AAC and try again
            transcode_cmd = ['ffmpeg', '-y', '-i', source_path, '-c:a', 'aac', '-b:a', '160k', '-vn', str(temp_audio_file_aac)]
            self._execute_ffmpeg_command(transcode_cmd)

            merge_cmd_aac = ['ffmpeg', '-y', '-i', no_audio_path, '-i', str(temp_audio_file_aac), '-c', 'copy', final_path]
            try:
                self._execute_ffmpeg_command(merge_cmd_aac)
                print("Audio transferred successfully after transcoding to AAC.")
            except subprocess.CalledProcessError:
                print("Error: Audio transfer failed completely. Output video will have no audio.")
                # If all fails, rename the no-audio video to the final name
                shutil.move(no_audio_path, final_path)
        finally:
            # 4. Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            if Path(no_audio_path).exists():
                 Path(no_audio_path).unlink()

    def _get_video_info(self, video_path: str):
        """
        Gets video information using ffprobe, with a fallback for counting frames.
        """
        # Command to get stream info including the potentially unreliable nb_frames
        info_command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = self._execute_ffmpeg_command(info_command)
        output = result.stdout.strip().split('\n')
        
        # Check for audio stream
        audio_command = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        try:
            audio_result = self._execute_ffmpeg_command(audio_command)
            has_audio = 'audio' in audio_result.stdout
        except subprocess.CalledProcessError:
            has_audio = False # No audio stream found
        
        # FIX 1: Correct the unpacking order and make it robust against malformed output.
        if len(output) < 5 or not all(output):
             raise RuntimeError(f"Could not retrieve all required video stream information from {video_path}. Ffprobe output: {output}")
        print('output', output)
        codec_type, width, height, frame_rate_str, nb_frames_str = output

        # FIX 2: Handle 'N/A' for frame count by manually counting frames.
        try:
            total_frames = int(nb_frames_str)
        except Exception as e:
            traceback.print_exc()
            print("Warning: 'nb_frames' not found in metadata. Counting frames manually (this may take a moment)...")
            count_command = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_frames', '-show_entries', 'stream=nb_read_frames',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            try:
                count_result = self._execute_ffmpeg_command(count_command)
                total_frames = int(count_result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError) as e:
                raise RuntimeError(f"Failed to count frames for {video_path}. Error: {e}") from e
        print('total_frames', total_frames)

        try:
            num, den = map(int, frame_rate_str.split('/'))
            fps = num / den
        except (ValueError, ZeroDivisionError):
            # Fallback for frame rates not in fraction format or if denominator is zero
            fps = float(frame_rate_str)
            
        return {
            'width': int(width), 'height': int(height), 'fps': fps,
            'total_frames': total_frames, 'has_audio': has_audio
        }

    def process(self,
                input_video_path: str,
                multiplier: int = 2,
                scale: float = 1.0,
                output_dir: str = 'output/',
                scene_change_threshold: float = 0.2,
                static_threshold: float = 0.996) -> str:
        """
        Main processing function to interpolate a video to a higher frame rate.

        Args:
            input_video_path (str): Path to the input video file.
            multiplier (int): The factor to multiply the frame rate (must be a power of 2).
            scale (float): The upscale factor for the RIFE model (1.0 = no spatial upscale).
            output_dir (str): Directory to save the output video.
            scene_change_threshold (float): SSIM threshold below which a scene change is detected.
            static_threshold (float): SSIM threshold above which frames are considered static.

        Returns:
            str: The file path of the generated video.
        """
        if not (multiplier > 0 and (multiplier & (multiplier - 1) == 0)):
             raise ValueError("Multiplier must be a power of 2 (e.g., 2, 4, 8).")
        
        # --- 1. Setup Environment and Get Video Info ---
        video_path = Path(input_video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        temp_base_dir = output_path / f"temp_{video_path.stem}_{os.getpid()}"
        input_frames_dir = temp_base_dir / "input_frames"
        output_frames_dir = temp_base_dir / "output_frames"
        input_frames_dir.mkdir(parents=True, exist_ok=True)
        output_frames_dir.mkdir(parents=True, exist_ok=True)
        
        info = self._get_video_info(str(video_path))
        input_fps = info['fps']
        output_fps = input_fps * multiplier

        if (info['width'] > 2000 or info['height'] > 2000) and scale == 1.0:
            print("Warning: High-resolution video detected. Consider using scale=0.5 to reduce VRAM usage.")

        # --- 2. Extract Frames from Input Video ---
        print(f"Extracting {info['total_frames']} frames from video...")
        extract_command = [
            'ffmpeg', '-i', str(video_path), '-vsync', '0', str(input_frames_dir / 'frame_%08d.png')
        ]
        self._execute_ffmpeg_command(extract_command)
        
        input_frame_paths = sorted(list(input_frames_dir.glob('*.png')))
        if not input_frame_paths:
            raise ValueError("FFmpeg failed to extract any frames. Check the video file integrity.")

        # --- 3. Interpolate Frames with Scene/Static Detection ---
        pbar = tqdm(total=len(input_frame_paths) - 1, desc="Interpolating Frames")
        
        frame_count = 0
        last_img_tensor = None
        
        for i in range(len(input_frame_paths) - 1):
            img0_pil = Image.open(input_frame_paths[i]).convert("RGB")
            img1_pil = Image.open(input_frame_paths[i+1]).convert("RGB")
            
            # Write the first frame of the pair
            img0_pil.save(output_frames_dir / f"frame_{frame_count:08d}.png")
            frame_count += 1
            
            # Preprocess and pad images, re-using the previous tensor to save time
            if i == 0:
                img0_tensor, _ = self._pad_image(self._preprocess(img0_pil))
            else:
                img0_tensor = last_img_tensor
            
            img1_tensor, orig_size1 = self._pad_image(self._preprocess(img1_pil))
            last_img_tensor = img1_tensor
            
            ssim_score = self._calculate_ssim(img0_tensor, img1_tensor)
            
            interpolated_pils = []
            if ssim_score > static_threshold:
                # Static scene: duplicate the first frame
                for _ in range(multiplier - 1):
                    interpolated_pils.append(img0_pil)
            elif ssim_score < scene_change_threshold:
                # Scene change: duplicate the first frame to avoid ghosting
                for _ in range(multiplier - 1):
                    interpolated_pils.append(img0_pil)
            else:
                # Normal interpolation
                interpolated_tensors = self._generate_interpolated_frames(img0_tensor, img1_tensor, multiplier, scale)
                for t in interpolated_tensors:
                    w_orig, h_orig = orig_size1
                    pil_frame = self._postprocess(t[:, :, :h_orig, :w_orig])
                    interpolated_pils.append(pil_frame)
            
            # Save the generated intermediate frames
            for pil_frame in interpolated_pils:
                pil_frame.save(output_frames_dir / f"frame_{frame_count:08d}.png")
                frame_count += 1
            
            pbar.update(1)
        
        # Save the very last frame of the video
        Image.open(input_frame_paths[-1]).convert("RGB").save(output_frames_dir / f"frame_{frame_count:08d}.png")
        pbar.close()

        # --- 4. Encode New Video and Handle Audio ---
        video_filename = f"{video_path.stem}_x{multiplier}_{int(round(output_fps))}fps.mp4"
        final_video_filepath = str(output_path / video_filename)
        temp_video_filepath = str(temp_base_dir / "video_no_audio.mp4")

        print(f"Encoding video to {final_video_filepath} at {output_fps:.2f} FPS...")
        encode_command = [
            'ffmpeg', '-y', '-framerate', str(output_fps),
            '-i', str(output_frames_dir / 'frame_%08d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
            '-preset', 'fast', '-an', temp_video_filepath # Encode without audio first
        ]
        self._execute_ffmpeg_command(encode_command)
        
        if info['has_audio']:
            self._transfer_audio(str(video_path), temp_video_filepath, final_video_filepath)
        else:
            print("Source video has no audio stream. Skipping audio transfer.")
            shutil.move(temp_video_filepath, final_video_filepath)

        # --- 5. Cleanup ---
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_base_dir)
        print("Video processing complete.")
        return final_video_filepath


# --- UNIT TEST ---
if __name__ == '__main__':
    print("--- Running RIFE Unit Test for Video Interpolation ---")

    # 1. Setup Test Environment
    TEST_DIR = Path("./rife_test_env")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir()

    (TEST_DIR / "train_log").mkdir()
    MODEL_DIR = TEST_DIR / "model"
    MODEL_DIR.mkdir()
    INPUT_DIR = TEST_DIR / "test_input"
    INPUT_DIR.mkdir()
    OUTPUT_DIR = TEST_DIR / "test_output"
    OUTPUT_DIR.mkdir()

    try:
        # Create dummy model files and checkpoint
        print("Creating dummy model files and checkpoint...")
        (TEST_DIR / "train_log" / "__init__.py").touch()
        (MODEL_DIR / "__init__.py").touch()
        project_root = Path(__file__).resolve().parent
        # Copy necessary source files
        shutil.copy(project_root / "train_log/IFNet_HDv3.py", TEST_DIR / "train_log/")
        shutil.copy(project_root / "train_log/RIFE_HDv3.py", TEST_DIR / "train_log/")
        shutil.copy(project_root / "train_log/refine.py", TEST_DIR / "train_log/")
        shutil.copy(project_root / "model/warplayer.py", MODEL_DIR)
        
        # Create a dummy pytorch_msssim.py with a basic SSIM implementation for testing
        (MODEL_DIR / "pytorch_msssim.py").write_text("""
import torch
def ssim_matlab(img1, img2):
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    C1, C2 = (0.01 * 1)**2, (0.03 * 1)**2
    ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return ssim_num / ssim_den
""")
        (MODEL_DIR / "loss.py").write_text("class EPE: pass\nclass SOBEL: pass\n")
        
        sys.path.insert(0, str(TEST_DIR))
        from train_log.RIFE_HDv3 import Model as TestModel
        test_model_instance = TestModel()
        torch.save(test_model_instance.flownet.state_dict(), TEST_DIR / 'train_log/flownet.pkl')
        sys.path.pop(0)

        # Create a short test video with scene changes and static parts
        print("Creating dummy input video...")
        Image.new('RGB', (128, 128), color='red').save(INPUT_DIR / "f01.png")   # Start
        Image.new('RGB', (128, 128), color='blue').save(INPUT_DIR / "f02.png")  # Scene change
        Image.new('RGB', (128, 128), color='blue').save(INPUT_DIR / "f03.png")  # Static
        Image.new('RGB', (128, 128), color='green').save(INPUT_DIR / "f04.png") # Scene change
        
        test_video_path = str(INPUT_DIR / "test_video.mp4")
        test_input_fps = 1
        create_video_command = [
            'ffmpeg', '-y', '-framerate', str(test_input_fps),
            '-i', str(INPUT_DIR / 'f%02d.png'), '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p', test_video_path
        ]
        subprocess.run(create_video_command, check=True, capture_output=True)
        assert Path(test_video_path).exists(), "Test video creation failed."

        # 2. Run the Test
        print("\n--- Instantiating and Processing ---")
        # Need to re-add test dir to path for the Rife class to import its modules
        sys.path.insert(0, str(TEST_DIR))
        rife_interpolator = Rife(model_dir=str(TEST_DIR / 'train_log'), use_fp16=True)
        sys.path.pop(0)
        
        test_multiplier = 4
        output_video_path_str = rife_interpolator.process(
            input_video_path=test_video_path,
            multiplier=test_multiplier,
            output_dir=str(OUTPUT_DIR)
        )
        
        # 3. Assertions
        print("\n--- Verifying Results ---")
        output_video_path = Path(output_video_path_str)
        assert output_video_path.exists(), f"Output video file was not created: {output_video_path}"
        assert output_video_path.is_file(), "Output path is not a file."
        print(f"✅ Success: Video generated at '{output_video_path}'")

        output_info = rife_interpolator._get_video_info(str(output_video_path))
        
        # Original frames = 4. Pairs = 3. Interpolated frames per pair = 3.
        # Total = 4 (original) + 3 (pairs) * 3 (interp) = 13 frames.
        expected_frames = 4 + (4 - 1) * (test_multiplier - 1)
        assert output_info['total_frames'] == expected_frames, \
            f"Expected {expected_frames} frames, but found {output_info['total_frames']}."
        print(f"✅ Success: Video contains the correct number of frames ({expected_frames}).")
        
        expected_fps = test_input_fps * test_multiplier
        assert abs(output_info['fps'] - expected_fps) < 0.01, \
            f"Expected {expected_fps} FPS, but found {output_info['fps']:.2f}."
        print(f"✅ Success: Video has the correct frame rate ({expected_fps} FPS).")

    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 4. Cleanup
        print("\n--- Cleaning up test environment ---")
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)
            print(f"Removed test directory: {TEST_DIR}")
        print("--- Unit Test Finished ---")
