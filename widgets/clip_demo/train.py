# train.py

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import argparse

from model_clip import CLIPMultiAttributeModel, ATTRIBUTE_NAMES

# 为 ResNet-50 预训练模型定义的图像归一化
# [0, 255] uint8 -> [0, 1] float -> normalized float
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train(
    data_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_dir: str,
    log_dir: str,
    save_every: int,
    resume_from: str = None
):
    """主训练函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告: CUDA 不可用，将在 CPU 上训练。速度会非常慢。", file=sys.stderr)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. 数据加载
    print("正在加载数据...")
    try:
        data = np.load(data_path)
        images = torch.from_numpy(data['images']).permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        attributes = torch.from_numpy(data['attributes']).float()
        dataset = TensorDataset(images, attributes)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(f"数据加载完毕。共 {len(dataset)} 张图片。")
    except FileNotFoundError:
        print(f"错误: 数据文件未找到 at {data_path}", file=sys.stderr)
        sys.exit(1)

    # 2. 模型、优化器
    model = CLIPMultiAttributeModel().to(device)
    # 仅优化需要梯度的参数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"正在从 checkpoint 加载: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"成功加载，将从 epoch {start_epoch} 继续。")

    # 3. TensorBoard 和混合精度
    writer = SummaryWriter(log_dir)
    # bfloat16 不需要 GradScaler，但 autocast 仍是必须的
    # scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"使用混合精度类型: {dtype}")

    # 4. 训练循环
    model.train()
    total_steps = 0
    for epoch in range(start_epoch, epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for i, (batch_images, batch_attributes) in enumerate(pbar):
            batch_images = batch_images.to(device)
            batch_attributes = batch_attributes.to(device)

            # 归一化图像
            batch_images = IMAGE_TRANSFORM(batch_images)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=dtype, enabled=(device == "cuda")):
                image_features, attr_features = model(batch_images, batch_attributes)
                loss = model.calculate_loss(image_features, attr_features)
            
            # 使用 bfloat16 时，scaler 不是必需的
            loss.backward()
            optimizer.step()
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if total_steps % 20 == 0:
                writer.add_scalar('Loss/train', loss.item(), total_steps)
                pbar.set_postfix({"loss": loss.item()})
            
            total_steps += 1
        
        # 保存 checkpoint
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint 已保存至: {checkpoint_path}")
    
    writer.close()
    print("训练完成。")


# ================== Unit Test / Command Line Runner ==================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练多属性 CLIP 模型")
    parser.add_argument('--data_path', type=str, default='celeba_cache_64x64.npz', help='CelebA npz 文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoints 保存目录')
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard 日志的根目录')
    parser.add_argument('--save_every', type=int, default=1, help='每隔多少 epoch 保存一次')
    parser.add_argument('--resume_from', type=str, default=None, help='从指定 checkpoint 继续训练')
    parser.add_argument('--test_run', action='store_true', help='运行一个快速的单元测试')
    parser.add_argument('--run_name', type=str, default=None, help='实验的特定名称，用于命名目录')

    args = parser.parse_args()

    if args.test_run:
        print("正在运行单元测试...")
        # 创建假的 npz 数据
        test_data_path = "test_data.npz"
        if not os.path.exists(test_data_path):
            dummy_images = np.random.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8)
            dummy_attrs = np.random.randint(0, 2, (20, 40)).astype(np.float32)
            np.savez(test_data_path, images=dummy_images, attributes=dummy_attrs)
        
        # 运行一小轮训练
        train(
            data_path=test_data_path,
            epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            checkpoint_dir='test_checkpoints',
            log_dir='test_runs',
            save_every=1
        )
        print("单元测试完成。检查 'test_checkpoints' 和 'test_runs' 目录。")
        # 清理
        import shutil
        os.remove(test_data_path)
        shutil.rmtree('test_checkpoints')
        shutil.rmtree('test_runs')
    else:
        # 创建一个基于超参数的唯一实验名称
        if args.run_name:
            experiment_name = args.run_name
        else:
            experiment_name = f"e{args.epochs}_b{args.batch_size}_lr{args.lr}_se{args.save_every}"

        # 构建完整的日志和 checkpoint 路径
        final_checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_name)
        final_log_dir = os.path.join(args.log_dir, experiment_name)

        # 正常训练
        train(
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_dir=final_checkpoint_dir,
            log_dir=final_log_dir,
            save_every=args.save_every,
            resume_from=args.resume_from
        )
