import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# --- Time Embedding ---
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

# --- NEW: Attention Mechanism for Conditioning ---
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj=True, out_proj=True):
        super().__init__()
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_cross, d_embed)
        self.v_proj = nn.Linear(d_cross, d_embed)
        self.out_proj = nn.Linear(d_embed, d_embed)

    def forward(self, x, context):
        # x: (Batch, Seq_Len, Dim_Embed)
        # context: (Batch, Seq_Len_Context, Dim_Context)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)

        return self.out_proj(output)


# --- U-Net Components (MODIFIED) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_classes=None, class_emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        self.time_emb = nn.Linear(time_channels, out_channels)

        # --- NEW: Conditional Part ---
        self.class_emb = None
        self.attention = None
        if n_classes is not None and class_emb_dim is not None:
            self.attention = CrossAttention(n_heads=4, d_embed=out_channels, d_cross=class_emb_dim)


    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        
        time_transformed = self.time_emb(self.act2(t))
        h += time_transformed[:, :, None, None]
        
        # --- NEW: Apply cross-attention if context is provided ---
        if context is not None and self.attention is not None:
            n, c, height, width = h.shape
            h_attn = h.view(n, c, height * width).permute(0, 2, 1) # (N, H*W, C)
            context = context.unsqueeze(1) # (N, 1, D_cross)
            attn_out = self.attention(h_attn, context)
            attn_out = attn_out.permute(0, 2, 1).view(n, c, height, width)
            h = h + attn_out

        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_classes=None, class_emb_dim=None):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, n_classes, class_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None):
        x = self.res(x, t, context)
        skip_connection = x
        x = self.downsample(x)
        return x, skip_connection

class UpBlock(nn.Module):
    def __init__(self, x_channels: int, skip_channels: int, out_channels: int, time_channels: int, n_classes=None, class_emb_dim=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(x_channels, x_channels, kernel_size=4, stride=2, padding=1)
        self.res = ResidualBlock(x_channels + skip_channels, out_channels, time_channels, n_classes, class_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None):
        x = self.upsample(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat((skip, x), dim=1)
        x = self.res(x, t, context)
        return x

# --- U-Net Model (MODIFIED) ---
class UNet(nn.Module):
    def __init__(self, image_channels: int = 3, n_channels: int = 32,
                 ch_mults: tuple = (1, 2, 2, 4),
                 time_emb_dim: int = 128,
                 # --- NEW: Conditioning params ---
                 n_classes: int = None,
                 class_emb_dim: int = None,
                 cond_drop_prob: float = 0.1):
        super().__init__()

        self.image_channels = image_channels
        self.n_channels = n_channels
        self.time_emb_dim = time_emb_dim
        self.n_classes = n_classes
        self.class_emb_dim = class_emb_dim
        self.cond_drop_prob = cond_drop_prob

        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # --- NEW: Class/Attribute embedding ---
        if n_classes is not None and class_emb_dim is not None:
             # Using a Linear layer for multi-hot attribute vectors
            self.class_embedding = nn.Linear(n_classes, class_emb_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(self.image_channels, n_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        current_channels = n_channels
        for i, mult in enumerate(ch_mults):
            out_ch = n_channels * mult
            self.down_blocks.append(DownBlock(current_channels, out_ch, time_emb_dim, n_classes, class_emb_dim))
            current_channels = out_ch
        
        # Bottleneck
        self.mid_res1 = ResidualBlock(current_channels, current_channels, time_emb_dim, n_classes, class_emb_dim)
        self.mid_res2 = ResidualBlock(current_channels, current_channels, time_emb_dim, n_classes, class_emb_dim)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            upblock_x_channels = current_channels
            upblock_skip_channels = n_channels * mult
            upblock_out_channels = n_channels * mult
            
            self.up_blocks.append(UpBlock(upblock_x_channels, upblock_skip_channels, upblock_out_channels, time_emb_dim, n_classes, class_emb_dim))
            current_channels = upblock_out_channels

        self.final_norm = nn.GroupNorm(8, current_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(current_channels, self.image_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, class_labels: torch.Tensor = None):
        t_emb = self.time_embedding(time)
        
        # --- NEW: Get context from class labels ---
        context = None
        if self.n_classes is not None and class_labels is not None:
            context = self.class_embedding(class_labels.float()) # Ensure labels are float

            # Classifier-Free Guidance (CFG) dropout during training
            if self.training and self.cond_drop_prob > 0:
                mask = torch.rand(class_labels.shape[0], device=x.device) > self.cond_drop_prob
                context = context * mask.unsqueeze(1) # (N, 1) * (N, D_emb)

        x = self.init_conv(x)
        
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb, context)
            skip_connections.append(skip)
            
        x = self.mid_res1(x, t_emb, context)
        x = self.mid_res2(x, t_emb, context)
        
        skip_connections = skip_connections[::-1]
        
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections[i]
            x = up_block(x, skip, t_emb, context)
            
        x = self.final_act(self.final_norm(x))
        x = self.final_conv(x)
        
        return x

# --- NEW: Discriminator for GAN Loss ---
class Discriminator(nn.Module):
    def __init__(self, image_channels=3, n_channels=64, ch_mults=(1, 2, 4, 8), time_emb_dim=128, n_classes=None, class_emb_dim=None):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        if n_classes is not None:
            self.class_embedding = nn.Linear(n_classes, class_emb_dim)
            embedding_dim = time_emb_dim + class_emb_dim
        else:
            self.class_embedding = None
            embedding_dim = time_emb_dim

        self.init_conv = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)
        
        layers = []
        current_channels = n_channels
        for mult in ch_mults:
            out_ch = n_channels * mult
            layers.append(nn.Conv2d(current_channels, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GroupNorm(8, out_ch))
            layers.append(nn.SiLU())
            current_channels = out_ch
        
        self.encoder = nn.Sequential(*layers)

        self.final_conv = nn.Conv2d(current_channels, 1, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x, t, class_labels=None):
        t_emb = self.time_embedding(t)
        
        if self.class_embedding is not None and class_labels is not None:
            class_emb = self.class_embedding(class_labels.float())
            emb = torch.cat([t_emb, class_emb], dim=1)
        else:
            emb = t_emb
        
        x = self.init_conv(x)
        x = self.encoder(x)
        
        # We can add the embedding to the feature map before the final conv
        # Or just use it in the head. For simplicity, we can flatten and concat.
        x = self.final_conv(x)
        
        return x # Output is a logit map

# --- DDPM Scheduler & Noise Functions (MODIFIED SAMPLER) ---
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def get_ddpm_params(schedule):
    betas = schedule
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return {"betas": betas, "alphas_cumprod": alphas_cumprod, "sqrt_alphas_cumprod": sqrt_alphas_cumprod, "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod, "posterior_variance": posterior_variance, "sqrt_recip_alphas": torch.sqrt(1.0 / alphas)}

def q_sample(x_start, t, ddpm_params, noise=None):
    if noise is None: noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = ddpm_params["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x_t, t_tensor, t_idx, ddpm_params, class_labels=None):
    betas_t = ddpm_params["betas"][t_idx]
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"][t_idx]
    sqrt_recip_alphas_t = ddpm_params["sqrt_recip_alphas"][t_idx]
    
    model_output = model(x_t, t_tensor, class_labels)
    
    model_mean = sqrt_recip_alphas_t * (x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * model_output)
    
    posterior_variance_t = ddpm_params["posterior_variance"][t_idx]
    if t_idx == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, timesteps, ddpm_params, device, class_labels=None, guidance_scale=7.5):
    img = torch.randn(shape, device=device)
    imgs = []
    
    if class_labels is not None:
        # Create unconditional context (a tensor of zeros for the attributes)
        uncond_class_labels = torch.zeros_like(class_labels, device=device)

    for i in reversed(range(0, timesteps)):
        t_tensor = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        if class_labels is not None:
            # Predict noise with conditioning
            pred_noise_cond = model(img, t_tensor, class_labels)
            # Predict noise without conditioning
            pred_noise_uncond = model(img, t_tensor, uncond_class_labels)
            # Combine using CFG formula
            model_output = (1 + guidance_scale) * pred_noise_cond - guidance_scale * pred_noise_uncond
        else: # Unconditional generation
            model_output = model(img, t_tensor, None)

        # --- Denoise one step using the combined prediction ---
        betas_t = ddpm_params["betas"][i]
        sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"][i]
        sqrt_recip_alphas_t = ddpm_params["sqrt_recip_alphas"][i]
        
        model_mean = sqrt_recip_alphas_t * (img - (betas_t / sqrt_one_minus_alphas_cumprod_t) * model_output)
        
        posterior_variance_t = ddpm_params["posterior_variance"][i]
        if i == 0:
            img = model_mean
        else:
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        if i % (timesteps//10) == 0 or i < 10:
            imgs.append(img.cpu())
            
    return img.cpu(), imgs
