import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple

# --- Time Embedding (Unchanged from original) ---
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int): # Here n_channels is the time_emb_dim
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
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

# --- DiT Components ---

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B C H_p W_p -> B C (H_p*W_p) -> B (H_p*W_p) C
        return x

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, time_emb_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) # Ensure batch_first=True
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # AdaLN-Zero modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * hidden_size, bias=True) # 2 for norm1 (shift,scale), 2 for attn (not used here directly), 2 for norm2
        )

    def forward(self, x, c): # c is the time_embedding
        # c: (B, time_emb_dim)
        # x: (B, num_patches, hidden_size)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # MSA part
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        # PyTorch MHA expects (Query, Key, Value)
        # For self-attention, Query=Key=Value=x_norm1
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out # DiT applies gate after attention

        # MLP part
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out # DiT applies gate after MLP
        
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, time_emb_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# --- DiT Model (replaces UNet) ---
class UNet(nn.Module): # Keeping class name UNet as requested
    def __init__(self, 
                 image_channels: int = 1, 
                 input_size: int = 48, # e.g. 28 for 28x28 MNIST
                 patch_size: int = 4,   # e.g. 4 for 28x28 -> 7x7 patches
                 hidden_size: int = 256, # Renamed from n_channels, new meaning
                 depth: int = 12,       # Number of DiT blocks
                 num_heads: int = 4,    # Number of attention heads
                 mlp_ratio: float = 4.0,
                 time_emb_dim: int = 128):
        super().__init__()
        self.image_channels = image_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size # This is the transformer embedding dimension
        self.num_heads = num_heads
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim) # This output dim must match DiTBlock's expectation

        # Patchification and Positional Embedding
        self.patch_embed = PatchEmbed(img_size=input_size, patch_size=patch_size, in_chans=image_channels, embed_dim=hidden_size)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        
        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, time_emb_dim, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # Final layer to reconstruct image
        self.final_layer = FinalLayer(hidden_size, patch_size, image_channels, time_emb_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding and patch embedding
        nn.init.normal_(self.pos_embed, std=.02)
        # Initialize NNs
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # Typically Kaiming for Conv2d
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu') # Or SiLU if more appropriate
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine: # If true, LayerNorm has learnable params
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, N, P*P*C_out)
        return: (B, C_out, H, W)
        """
        B = x.shape[0]
        P = self.patch_size
        C_out = self.image_channels
        H_p = W_p = self.input_size // P # Number of patches in H/W dimension
        assert H_p * W_p == x.shape[1] # N = H_p * W_p

        x = x.reshape(shape=(B, H_p, W_p, P, P, C_out))
        x = torch.einsum('bhwpqc->bchpwq', x) # Permute to (B, C_out, H_p, P, W_p, Q)
        imgs = x.reshape(shape=(B, C_out, H_p * P, W_p * P)) # (B, C_out, H, W)
        return imgs

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        # x: (batch_size, image_channels, H, W)
        # time: (batch_size,)
        
        t_emb = self.time_embedding(time) # (batch_size, time_emb_dim)
        
        # Patchify and add positional embedding
        x_patches = self.patch_embed(x)  # (B, num_patches, hidden_size)
        x_patches = x_patches + self.pos_embed # Broadcasting (1, num_patches, hidden_size)
        
        # Apply DiT blocks
        for block in self.blocks:
            x_patches = block(x_patches, t_emb)
            
        # Apply final layer and unpatchify
        output_patches = self.final_layer(x_patches, t_emb) # (B, num_patches, patch_size*patch_size*image_channels)
        output_image = self.unpatchify(output_patches)     # (B, image_channels, H, W)
        
        return output_image


# --- DDPM Scheduler & Noise Functions (Unchanged) ---
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
    
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance
    }

def q_sample(x_start, t, ddpm_params, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Ensure ddpm_params tensors are on the same device as t for indexing
    device = t.device
    sqrt_alphas_cumprod_t = ddpm_params["sqrt_alphas_cumprod"].to(device)[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"].to(device)[t].view(-1, 1, 1, 1)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x_t, t_tensor, t_idx, ddpm_params):
    # model predicts noise_pred given x_t and t_idx
    # t_tensor is (batch_size,) tensor with current timestep index
    device = x_t.device # Ensure all calculations happen on the same device

    # Move relevant ddpm_params to device for indexing and calculations
    betas_t = ddpm_params["betas"].to(device)[t_idx]
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"].to(device)[t_idx]
    sqrt_recip_alphas_t = 1.0 / torch.sqrt(1.0 - betas_t)
    
    model_output = model(x_t, t_tensor) # model_output is predicted noise epsilon_theta
    
    coeff = (betas_t / sqrt_one_minus_alphas_cumprod_t)
    # Reshape for broadcasting if t_idx is scalar (common) or tensor
    if coeff.ndim == 0: # scalar
        coeff = coeff.view(1,1,1,1)
        sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(1,1,1,1)
    else: # tensor for batched t_idx (not typical in standard p_sample loop)
         coeff = coeff.view(-1,1,1,1)
         sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(-1,1,1,1)


    model_mean = sqrt_recip_alphas_t * (x_t - coeff * model_output)
    
    posterior_variance_t = ddpm_params["posterior_variance"].to(device)[t_idx]
    if t_idx == 0: # For the last step, t_idx becomes 0
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        if posterior_variance_t.ndim == 0:
            posterior_variance_t = posterior_variance_t.view(1,1,1,1)
        else:
            posterior_variance_t = posterior_variance_t.view(-1,1,1,1)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, timesteps, ddpm_params, device):
    img = torch.randn(shape, device=device)
    imgs = []

    # Ensure all ddpm_params are on the target device beforehand, or handle inside p_sample
    # For simplicity, let's assume ddpm_params can be passed as CPU tensors and p_sample handles device movement
    
    for i in reversed(range(0, timesteps)):
        t_tensor = torch.full((shape[0],), i, device=device, dtype=torch.long)
        # p_sample expects t_idx as a scalar Python int for indexing within the loop
        img = p_sample(model, img, t_tensor, i, ddpm_params)
        if i % (timesteps//10) == 0 or i < 10: # Log a few images
            imgs.append(img.cpu().clone()) # Use .clone() to avoid all appended tensors pointing to the same memory
    return img.cpu(), imgs

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # General parameters for model instantiation test (can be different from q_sample test)
    model_test_batch_size = 2 
    image_size = 16 
    patch_size = 4  
    image_channels = 1
    
    dit_model = UNet(
        image_channels=image_channels,
        input_size=image_size,      
        patch_size=patch_size,      
        hidden_size=64,             
        depth=3,                    
        num_heads=4,                
        time_emb_dim=128            
    ).to(device)
    
    dummy_x = torch.randn(model_test_batch_size, image_channels, image_size, image_size, device=device)
    dummy_t = torch.randint(0, 1000, (model_test_batch_size,), device=device).float()
    
    print(f"Input image shape: {dummy_x.shape}")
    predicted_noise = dit_model(dummy_x, dummy_t)
    print("DiT UNet output shape:", predicted_noise.shape)
    assert predicted_noise.shape == dummy_x.shape, \
        f"Output shape {predicted_noise.shape} does not match input shape {dummy_x.shape}"

    # Test DDPM params
    T = 1000
    betas = linear_beta_schedule(T) 
    params_cpu = get_ddpm_params(betas) 
    print("DDPM params keys:", params_cpu.keys())
    assert params_cpu["betas"].shape == (T,)

    # Test q_sample
    print("Testing q_sample...")
    # Define timesteps for this specific test
    t_indices_for_q_sample = torch.tensor([0, T // 2, T - 1], dtype=torch.long, device=device)
    num_q_sample_instances = len(t_indices_for_q_sample) # This will be 3
    
    # Create an x_start tensor with a batch size matching the number of timesteps
    x_start_for_q_sample_test = torch.randn(num_q_sample_instances, image_channels, image_size, image_size, device=device)
    
    # Now, x_start_for_q_sample_test.shape[0] (3) == len(t_indices_for_q_sample) (3)
    x_t_test = q_sample(x_start_for_q_sample_test, t_indices_for_q_sample, params_cpu)
    print("q_sample output shape:", x_t_test.shape)
    # The assertion should reflect the number of instances used for this q_sample test
    assert x_t_test.shape == (num_q_sample_instances, image_channels, image_size, image_size)
    
    # Test p_sample_loop (basic call)
    print("Testing p_sample_loop...")
    # p_sample_loop uses a batch size of 1 for the generated image shape
    p_loop_batch_size = 1
    final_img, _ = p_sample_loop(dit_model, (p_loop_batch_size, image_channels, image_size, image_size), T, params_cpu, device)
    print("p_sample_loop output shape:", final_img.shape)
    assert final_img.shape == (p_loop_batch_size, image_channels, image_size, image_size)

    print("Basic model_dit.py tests passed.")
