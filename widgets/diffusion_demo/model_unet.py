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
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = nn.SiLU() # Swish activation
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # (Refer to Attention is All You Need paper for details)
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

# --- U-Net Components ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        # Time embedding layer
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        
        # Add time embedding
        time_transformed = self.time_emb(self.act2(t)) # use SiLU for time emb too
        h += time_transformed[:, :, None, None] # Reshape for broadcasting
        
        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) # Or MaxPool2d

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        skip_connection = x # Store for skip connection
        x = self.downsample(x)
        return x, skip_connection

class UpBlock(nn.Module):
    def __init__(self, x_channels: int, skip_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        # x_channels: channels of the input tensor 'x' to be upsampled by self.upsample
        # skip_channels: channels of the skip connection tensor
        # out_channels: output channels of this UpBlock (i.e., of its ResidualBlock)
        
        # ConvTranspose2d: input is x_channels. Output channels are also x_channels to preserve them.
        self.upsample = nn.ConvTranspose2d(x_channels, x_channels, kernel_size=4, stride=2, padding=1)
        # ResidualBlock input channels = channels from upsample (x_channels) + channels from skip (skip_channels)
        self.res = ResidualBlock(x_channels + skip_channels, out_channels, time_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor):
        x = self.upsample(x)
        # Pad x if its spatial dimensions are smaller than skip's after upsampling (due to odd input sizes)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat((skip, x), dim=1) # Concatenate skip connection
        x = self.res(x, t)
        return x

# --- U-Net Model ---
class UNet(nn.Module):
    def __init__(self, image_channels: int = 1, n_channels: int = 32,
                 ch_mults: tuple = (1, 2, 2), # Multipliers for n_channels
                 time_emb_dim: int = 128):
        super().__init__()

        self.image_channels = image_channels
        self.n_channels = n_channels # Base number of channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(self.image_channels, n_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        current_channels = n_channels
        for i, mult in enumerate(ch_mults):
            out_ch = n_channels * mult
            self.down_blocks.append(DownBlock(current_channels, out_ch, time_emb_dim))
            current_channels = out_ch
        
        # Bottleneck
        self.mid_res1 = ResidualBlock(current_channels, current_channels, time_emb_dim)
        self.mid_res2 = ResidualBlock(current_channels, current_channels, time_emb_dim)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            upblock_x_channels = current_channels
            upblock_skip_channels = n_channels * mult
            upblock_out_channels = n_channels * mult
            
            self.up_blocks.append(UpBlock(upblock_x_channels, upblock_skip_channels, upblock_out_channels, time_emb_dim))
            current_channels = upblock_out_channels


        # Final layers
        # current_channels is now n_channels * ch_mults[0]
        # If ch_mults[0] is 1, then current_channels is n_channels.
        # If UNet is intended to always output n_channels before final_conv, ch_mults[0] must be 1.
        # The original code implies this.
        self.final_norm = nn.GroupNorm(8, current_channels) # Use actual current_channels
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(current_channels, self.image_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor):
        # x: (batch_size, image_channels, H, W)
        # time: (batch_size,)
        
        t_emb = self.time_embedding(time) # (batch_size, time_emb_dim)
        
        x = self.init_conv(x) # (batch_size, n_channels, H, W)
        
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb)
            skip_connections.append(skip)
            
        x = self.mid_res1(x, t_emb)
        x = self.mid_res2(x, t_emb)
        
        skip_connections = skip_connections[::-1] # Reverse for upsampling
        
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections[i]
            x = up_block(x, skip, t_emb)
            
        x = self.final_act(self.final_norm(x))
        x = self.final_conv(x)
        
        return x

# --- DDPM Scheduler & Noise Functions ---
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def get_ddpm_params(schedule):
    betas = schedule
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance
    }

# Forward diffusion (adding noise)
def q_sample(x_start, t, ddpm_params, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = ddpm_params["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# Reverse diffusion (denoising one step)
@torch.no_grad()
def p_sample(model, x_t, t_tensor, t_idx, ddpm_params):
    # model predicts noise_pred given x_t and t_idx
    # t_tensor is (batch_size,) tensor with current timestep index
    
    betas_t = ddpm_params["betas"][t_idx]
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"][t_idx]
    # For alpha_t = 1 - beta_t, sqrt_recip_alphas_t = 1.0 / torch.sqrt(1.0 - betas_t)
    sqrt_recip_alphas_t = 1.0 / torch.sqrt(1.0 - betas_t) # Corrected reciprocal calculation
    
    # Equation 11 in DDPM paper:
    # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1-alpha_cumprod_t) * epsilon_theta(x_t, t)) + sigma_t * z
    model_output = model(x_t, t_tensor)
    
    # Ensure model_output is used correctly in the formula
    # The term (betas_t / sqrt_one_minus_alphas_cumprod_t) should be correctly shaped for broadcasting
    coeff = (betas_t / sqrt_one_minus_alphas_cumprod_t)
    # Reshape coeff to be (B, 1, 1, 1) if x_t is (B, C, H, W) and model_output is (B, C, H, W)
    # betas_t and sqrt_one_minus_alphas_cumprod_t are scalars if t_idx is scalar
    # If t_idx can be a tensor, then they would be tensors.
    # Assuming t_idx is a scalar (from loop `i`), betas_t and sqrt_one_minus_alphas_cumprod_t are scalars.
    # sqrt_recip_alphas_t is also scalar.

    model_mean = sqrt_recip_alphas_t * (
        x_t - coeff * model_output
    )
    
    posterior_variance_t = ddpm_params["posterior_variance"][t_idx]
    if t_idx == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        # Ensure posterior_variance_t is also handled for shape if t_idx can be tensor
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, timesteps, ddpm_params, device):
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        t_tensor = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t_tensor, i, ddpm_params)
        if i % (timesteps//10) == 0 or i < 10: # Log a few images
            imgs.append(img.cpu()) 
    return img.cpu(), imgs


if __name__ == '__main__':
    # A quick test of the UNet structure
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 2
    image_size = 14 # MNIST 14x14
    image_channels = 1
    
    # Test U-Net
    unet = UNet(image_channels=image_channels, n_channels=32, ch_mults=(1, 2), time_emb_dim=128).to(device)
    dummy_x = torch.randn(batch_size, image_channels, image_size, image_size, device=device)
    dummy_t = torch.randint(0, 1000, (batch_size,), device=device).float() # DDPM timesteps
    
    predicted_noise = unet(dummy_x, dummy_t)
    print("U-Net output shape:", predicted_noise.shape)
    assert predicted_noise.shape == dummy_x.shape

    # Test DDPM params
    T = 1000
    betas = linear_beta_schedule(T)
    params = get_ddpm_params(betas.to(device)) # Move params to device if operations need it
    print("DDPM params keys:", params.keys())
    assert params["betas"].shape == (T,)

    # Test q_sample
    x_start_test = torch.randn(batch_size, image_channels, image_size, image_size, device=device)
    t_test = torch.tensor([0, T//2, T-1], device=device) # Test for a few timesteps
    
    # Ensure ddpm_params are on the same device as x_start_test and t_test for q_sample
    # This is implicitly handled if get_ddpm_params output tensors are moved to device,
    # or if the schedule itself is on the device.
    # For q_sample, ddpm_params tensors need to be on t.device.
    # Let's ensure params are on CPU for this test or x_start_test is on CPU if params are CPU.
    # For now, let's assume get_ddpm_params returns CPU tensors if schedule is CPU tensor.
    # And q_sample expects t and ddpm_params tensors to be on same device as x_start.
    # Let's make schedule on device.
    betas_dev = linear_beta_schedule(T).to(device)
    params_dev = get_ddpm_params(betas_dev)

    x_t_test = q_sample(x_start_test[:len(t_test)], t_test, params_dev)
    print("q_sample output shape:", x_t_test.shape)
    assert x_t_test.shape == (len(t_test), image_channels, image_size, image_size)
    
    # Test p_sample_loop (basic call)
    print("Testing p_sample_loop...")
    final_img, _ = p_sample_loop(unet, (1, image_channels, image_size, image_size), T, params_dev, device)
    print("p_sample_loop output shape:", final_img.shape)
    assert final_img.shape == (1, image_channels, image_size, image_size)

    print("Basic model_unet.py tests passed.")
