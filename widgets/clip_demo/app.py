# app.py

import os
import io
import base64
import torch
import numpy as np
import subprocess
import signal
import glob
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

from model_clip import CLIPMultiAttributeModel, ATTRIBUTE_NAMES
from train import IMAGE_TRANSFORM

app = Flask(__name__)

# --- 全局状态 ---
# 使用字典来存储进程信息，以支持多会话（虽然这里简化为单个进程）
training_process = {'proc': None, 'log_file': 'train.log'}

# --- 全局缓存 (用于测试) ---
# 缓存整个数据集的图像和预计算的 embeddings，避免重复加载和计算
DATA_CACHE = {'images': None, 'attributes': None}
EMBEDDING_CACHE = {'checkpoint_path': None, 'embeddings': None}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

# --- 数据加载辅助函数 ---
def load_full_dataset():
    """加载并缓存整个 CelebA 数据集"""
    if DATA_CACHE['images'] is None:
        print("首次加载数据集到内存...")
        try:
            data = np.load('celeba_cache_64x64.npz')
            # 仅加载部分数据以加快演示速度和减少内存占用
            # 您可以修改这里的 N 值
            N = 220000 
            DATA_CACHE['images'] = torch.from_numpy(data['images'][:N])
            DATA_CACHE['attributes'] = torch.from_numpy(data['attributes'][:N])
            print(f"已加载 {len(DATA_CACHE['images'])} 张图片和属性。")
        except FileNotFoundError:
            print("错误: celeba_cache_64x64.npz 未找到！")
            return False
    return True

# --- 路由 ---
@app.route('/')
def index():
    return render_template('index.html', attributes=ATTRIBUTE_NAMES)

@app.route('/start_train', methods=['POST'])
def start_train():
    load_full_dataset()
    if training_process['proc'] and training_process['proc'].poll() is None:
        return jsonify({'status': 'error', 'message': '一个训练任务已在运行中。'})

    config = request.json

    # 从配置中提取超参数以构建实验名称
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['lr']
    save_every = config['save_every']
    run_name = f"exp_e{epochs}_b{batch_size}_lr{lr}_se{save_every}"
    
    # 构建命令行参数
    cmd = [
        "python", "train.py",
        "--epochs", str(config['epochs']),
        "--batch_size", str(config['batch_size']),
        "--lr", str(config['lr']),
        "--save_every", str(config['save_every']),
        "--run_name", run_name,
    ]
    if config.get('resume_from'):
        cmd.extend(["--resume_from", config['resume_from']])

    # 使用 subprocess.Popen 将训练作为子进程启动
    # 将 stdout 和 stderr 重定向到日志文件
    with open(training_process['log_file'], 'w') as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    
    training_process['proc'] = proc
    return jsonify({'status': 'ok', 'message': '训练已开始。'})

@app.route('/stop_train', methods=['POST'])
def stop_train():
    proc_info = training_process['proc']
    if not proc_info or proc_info.poll() is not None:
        return jsonify({'status': 'error', 'message': '没有正在运行的训练任务。'})

    # 发送 SIGTERM 信号以优雅地终止进程（如果可能）
    # 在 Windows 上，这通常等同于 terminate()
    proc_info.send_signal(signal.SIGTERM) 
    try:
        proc_info.wait(timeout=10) # 等待 10 秒
    except subprocess.TimeoutExpired:
        proc_info.kill() # 如果无法优雅终止，则强制杀死

    training_process['proc'] = None
    return jsonify({'status': 'ok', 'message': '训练已停止。'})

@app.route('/train_status')
def train_status():
    proc = training_process['proc']
    if proc and proc.poll() is None:
        return jsonify({'status': 'running'})
    return jsonify({'status': 'stopped'})

@app.route('/train_log')
def train_log():
    try:
        # 安全地提供日志文件内容
        return send_from_directory('.', training_process['log_file'], mimetype='text/plain')
    except FileNotFoundError:
        return "日志文件尚未创建。", 404

@app.route('/get_checkpoints')
def get_checkpoints():
    checkpoint_dir = 'checkpoints'
    if not os.path.isdir(checkpoint_dir):
        return jsonify([])
    
    files = glob.glob(checkpoint_dir + '/*/*.pth')
    # 按修改时间排序，最新的在前面
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return jsonify(files)


@app.route('/test', methods=['POST'])
def test_model():
    data = request.json
    checkpoint_path = data.get('checkpoint')
    selected_attrs = data.get('attributes', [])
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return jsonify({'error': '请选择一个有效的 checkpoint。'}), 400
    if not selected_attrs:
        return jsonify({'error': '请至少选择一个属性。'}), 400
    if not load_full_dataset():
        return jsonify({'error': '服务器数据集文件缺失。'}), 500

    try:
        # --- 1. 加载模型 ---
        model = CLIPMultiAttributeModel().to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # --- 2. 预计算或从缓存加载所有图像的 embedding ---
        if EMBEDDING_CACHE.get('checkpoint_path') != checkpoint_path:
            print(f"为 checkpoint '{checkpoint_path}' 计算图像 embeddings...")
            all_images = DATA_CACHE['images'].to(DEVICE)
            all_img_embeddings = []
            with torch.no_grad():
                # 分批计算以避免 OOM
                for i in range(0, len(all_images), 128):
                    batch = all_images[i:i+128]
                    # 缓存中的图像是 (B, H, W, C) uint8.
                    # IMAGE_TRANSFORM (特别是 Normalize) 需要 (B, C, H, W) float.
                    batch = batch.permute(0, 3, 1, 2) # 维度转换: (B, H, W, C) -> (B, C, H, W)
                    batch = IMAGE_TRANSFORM(batch) # 应用转换 (float conversion, normalization)
                    with torch.cuda.amp.autocast(dtype=MODEL_DTYPE, enabled=(DEVICE=="cuda")):
                        img_embeds = model.image_encoder(batch.to(DEVICE))
                    all_img_embeddings.append(img_embeds.cpu())
            
            EMBEDDING_CACHE['embeddings'] = torch.cat(all_img_embeddings)
            EMBEDDING_CACHE['checkpoint_path'] = checkpoint_path
            print("Embeddings 计算并缓存完毕。")
        
        all_img_embeddings = EMBEDDING_CACHE['embeddings'].to(DEVICE)

        # --- 3. 计算目标属性的 embedding ---
        attr_vector = torch.zeros(len(ATTRIBUTE_NAMES))
        for attr in selected_attrs:
            attr_vector[ATTRIBUTE_NAMES.index(attr)] = 1.0
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=MODEL_DTYPE, enabled=(DEVICE=="cuda")):
                target_attr_embedding = model.attribute_encoder(attr_vector.unsqueeze(0).to(DEVICE))

        # --- 4. 计算相似度并排序 ---
        similarities = torch.nn.functional.cosine_similarity(target_attr_embedding, all_img_embeddings)
        top_k = torch.topk(similarities, k=12, largest=True) # 返回前 12 个

        # --- 5. 准备结果 ---
        results = []
        for score, idx in zip(top_k.values.cpu().tolist(), top_k.indices.cpu().tolist()):
            image_array = DATA_CACHE['images'][idx].numpy()
            img = Image.fromarray(image_array)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            results.append({
                'image': f'data:image/png;base64,{img_str}',
                'score': f'{score:.4f}'
            })

        return jsonify(results)

    except Exception as e:
        print(f"测试时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'服务器内部错误: {e}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
