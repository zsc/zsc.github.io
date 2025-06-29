# model_clip.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# CelebA 数据集定义的 40 个属性名称
# 这个列表的顺序必须与 .npz 文件中的属性列顺序一致
ATTRIBUTE_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
    'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
    'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

class AttributeEncoder(nn.Module):
    """
    将 40 维的多属性 one-hot/multi-hot 向量编码为 embedding。
    结构: Embedding (通过 Linear 实现) -> MLP
    """
    def __init__(self, num_attributes: int = 40, embedding_dim: int = 512):
        super().__init__()
        # 使用 Linear 层作为 embedding，输入 (B, 40)，权重 (40, dim)，输出 (B, dim)
        # 这等价于对每个激活属性的 embedding 向量求和
        self.embedding = nn.Linear(num_attributes, embedding_dim, bias=False)
        
        # 类似 Transformer FFN 的 MLP 结构
        self.mlp = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attributes (torch.Tensor): 形状为 (B, 40) 的属性张量
        Returns:
            torch.Tensor: 形状为 (B, embedding_dim) 的归一化 embedding
        """
        x = self.embedding(attributes.float())
        x = x + self.mlp(x) # Residual connection
        return F.normalize(x, p=2, dim=-1)


class ImageEncoder(nn.Module):
    """
    使用冻结的 ResNet-50 编码图像。
    将 64x64 图像放大到 224x224 后输入 ResNet。
    """
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        # 加载预训练的 ResNet-50
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # 替换最后的 fc 层以输出我们需要的 embedding 维度
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, embedding_dim)
        
        # 冻结除 fc 层外的所有参数
        for name, param in self.base_model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        # 定义图像放大操作
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): 形状为 (B, 3, 64, 64) 的图像张量
        Returns:
            torch.Tensor: 形状为 (B, embedding_dim) 的归一化 embedding
        """
        # 确认图像是 float 类型并放大
        images = self.upsample(images.float())
        features = self.base_model(images)
        return F.normalize(features, p=2, dim=-1)


class CLIPMultiAttributeModel(nn.Module):
    """
    组合图像和属性编码器的 CLIP 模型。
    """
    def __init__(self, num_attributes: int = 40, embedding_dim: int = 512):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.attribute_encoder = AttributeEncoder(num_attributes, embedding_dim)
        # logit_scale 是 CLIP 的可学习温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # np.log(1/0.07)

    def forward(self, images: torch.Tensor, attributes: torch.Tensor):
        img_features = self.image_encoder(images)
        attr_features = self.attribute_encoder(attributes)
        return img_features, attr_features

    def calculate_loss(self, image_features: torch.Tensor, attr_features: torch.Tensor):
        """
        计算对比损失 (InfoNCE loss)。
        """
        logit_scale = self.logit_scale.exp()
        # 计算图像和属性 embedding 之间的余弦相似度
        logits_per_image = logit_scale * image_features @ attr_features.t()
        logits_per_attr = logits_per_image.t()

        # 对称的交叉熵损失
        labels = torch.arange(len(image_features), device=image_features.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_a = F.cross_entropy(logits_per_attr, labels)
        return (loss_i + loss_a) / 2

# ================== Unit Test ==================
if __name__ == '__main__':
    print("Running model unit tests...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 测试 AttributeEncoder
    print("\nTesting AttributeEncoder...")
    attr_encoder = AttributeEncoder().to(device)
    dummy_attrs = torch.randint(0, 2, (4, 40)).float().to(device)
    attr_embeds = attr_encoder(dummy_attrs)
    assert attr_embeds.shape == (4, 512), f"AttributeEncoder output shape is {attr_embeds.shape}"
    print("AttributeEncoder passed.")

    # 2. 测试 ImageEncoder
    print("\nTesting ImageEncoder...")
    img_encoder = ImageEncoder().to(device)
    dummy_imgs = torch.randn(2, 3, 64, 64).to(device)
    img_embeds = img_encoder(dummy_imgs)
    assert img_embeds.shape == (2, 512), f"ImageEncoder output shape is {img_embeds.shape}"
    # 检查冻结是否生效
    for name, param in img_encoder.named_parameters():
        if 'fc' not in name:
            assert not param.requires_grad, f"Parameter {name} should be frozen."
        else:
            assert param.requires_grad, f"Parameter {name} should be trainable."
    print("ImageEncoder passed.")

    # 3. 测试完整的 CLIPMultiAttributeModel 和损失计算
    print("\nTesting CLIPMultiAttributeModel...")
    model = CLIPMultiAttributeModel().to(device)
    batch_size = 8
    dummy_imgs = torch.randn(batch_size, 3, 64, 64).to(device)
    dummy_attrs = torch.randint(0, 2, (batch_size, 40)).float().to(device)
    
    img_features, attr_features = model(dummy_imgs, dummy_attrs)
    assert img_features.shape == (batch_size, 512)
    assert attr_features.shape == (batch_size, 512)
    print("Forward pass passed.")

    loss = model.calculate_loss(img_features, attr_features)
    assert loss.item() > 0, "Loss should be a positive value"
    loss.backward()
    print("Loss calculation and backward pass passed.")

    # 检查梯度只在可训练参数上
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Trainable param {name} has no grad."
        else:
            assert param.grad is None, f"Frozen param {name} should not have grad."
    print("Gradient check passed.")

    print("\nAll unit tests passed successfully!")
