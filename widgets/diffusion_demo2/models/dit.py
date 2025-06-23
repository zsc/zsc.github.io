import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

# --- DiT Components ---
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) # Added batch_first=True
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential( # For modulating norms
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True) # scale and shift
        )

    def forward(self, x, c): # c is the conditional embedding (timestep)
        # Adaptive LayerNorm modulation
        shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
        # Reshape scale/shift for broadcasting: (B, 1, D) for (B, N, D) input x
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        
        # MSA block
        x_norm1 = self.norm1(x)
        x_norm1_modulated = (x_norm1 * (1 + scale_msa)) + shift_msa
        attn_out, _ = self.attn(x_norm1_modulated, x_norm1_modulated, x_norm1_modulated)
        x = x + attn_out

        # FFN block
        shift_ffn, scale_ffn = self.adaLN_modulation(c).chunk(2, dim=1) # Re-generate or use separate modulators
        shift_ffn = shift_ffn.unsqueeze(1)
        scale_ffn = scale_ffn.unsqueeze(1)

        x_norm2 = self.norm2(x)
        x_norm2_modulated = (x_norm2 * (1 + scale_ffn)) + shift_ffn
        mlp_out = self.mlp(x_norm2_modulated)
        x = x + mlp_out
        return x


class DiT(nn.Module):
    def __init__(self, latent_shape=(16, 6, 6), patch_size=2, in_channels=16, 
                 hidden_size=256, depth=4, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.latent_shape = latent_shape # C_latent, H_latent, W_latent from VAE
        self.patch_size = patch_size
        self.in_channels = in_channels # Should match C_latent
        self.out_channels = in_channels # Predict noise of same shape as input latent
        self.hidden_size = hidden_size
        
        C_latent, H_latent, W_latent = latent_shape
        assert H_latent % patch_size == 0 and W_latent % patch_size == 0, "Latent dimensions must be divisible by patch_size"
        
        self.num_patches = (H_latent // patch_size) * (W_latent // patch_size)
        self.patch_embed_dim = in_channels * (patch_size ** 2)

        self.x_embedder = nn.Linear(self.patch_embed_dim, hidden_size) # Patch projection
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, self.patch_embed_dim, bias=True) # Predict patches
        self.adaLN_modulation_final = nn.Sequential( # For final norm
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def unpatchify(self, x):
        # x: (B, num_patches, patch_embed_dim)
        # patch_embed_dim = C_latent * patch_size * patch_size
        C = self.in_channels
        P = self.patch_size
        H_latent, W_latent = self.latent_shape[1], self.latent_shape[2]
        H_patch, W_patch = H_latent // P, W_latent // P # Num patches in H, W dims
        
        # (B, H_patch * W_patch, C * P * P) -> (B, H_patch*W_patch, C, P, P)
        x = x.view(x.shape[0], H_patch, W_patch, C, P, P)
        # (B, C, H_patch, P, W_patch, P) using einops or permute+reshape
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous() # B, C, H_patch, P, W_patch, P
        # (B, C, H_patch*P, W_patch*P) = (B, C, H_latent, W_latent)
        images = x.view(x.shape[0], C, H_latent, W_latent)
        return images

    def forward(self, x_latent, t): # x_latent is (B, C_latent, H_latent, W_latent)
        B, C_l, H_l, W_l = x_latent.shape
        
        # Patchify input
        # (B, C_l, H_l, W_l) -> (B, C_l, H_patch, P, W_patch, P)
        x_patched = x_latent.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # (B, C_l, H_patch, W_patch, P, P)
        x_patched = x_patched.permute(0, 2, 3, 1, 4, 5).contiguous() # B, H_patch, W_patch, C_l, P, P
        # (B, H_patch*W_patch, C_l*P*P) = (B, num_patches, patch_embed_dim)
        x_patched_flat = x_patched.view(B, self.num_patches, -1)

        x_embedded = self.x_embedder(x_patched_flat) # (B, num_patches, hidden_size)
        x_embedded = x_embedded + self.pos_embed # Add positional embedding

        t_embedded = self.t_embedder(t) # (B, hidden_size)

        for block in self.blocks:
            x_embedded = block(x_embedded, t_embedded)
        
        # Final LayerNorm modulation
        shift_final, scale_final = self.adaLN_modulation_final(t_embedded).chunk(2, dim=1)
        shift_final = shift_final.unsqueeze(1)
        scale_final = scale_final.unsqueeze(1)

        x_final_norm = self.final_norm(x_embedded)
        x_final_norm_modulated = (x_final_norm * (1 + scale_final)) + shift_final
        x_out_patched = self.final_linear(x_final_norm_modulated) # (B, num_patches, patch_embed_dim)
        
        # Unpatchify to latent image shape
        predicted_noise_latent = self.unpatchify(x_out_patched) # (B, C_latent, H_latent, W_latent)
        return predicted_noise_latent

# Test for dit.py
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example: VAE encodes 48x48x3 image to 12x12x16 latent
    # (after 2 downsamples in VAE encoder, and VAE latent_dim chosen to match this structure for simplicity)
    # Or, if VAE latent_dim is flat (e.g. 256), reshape it to (B, C_l, H_l, W_l) before DiT
    # For this test, let's assume VAE output is already in a spatial latent form
    # If VAE encoder output is (B, 16, 12, 12) for example:
    latent_channels = 16 
    latent_h, latent_w = 12, 12 
    patch_size = 2 # DiT patch size
    
    # Check if dimensions are compatible
    if latent_h % patch_size != 0 or latent_w % patch_size != 0:
        print(f"Warning: Latent dimensions ({latent_h}x{latent_w}) not perfectly divisible by patch_size ({patch_size}). Adjusting for test.")
        # Adjust latent_h/w or patch_size for the test to work
        latent_h = (latent_h // patch_size) * patch_size
        latent_w = (latent_w // patch_size) * patch_size
        print(f"Adjusted latent dimensions for test: {latent_h}x{latent_w}")


    dit_model = DiT(
        latent_shape=(latent_channels, latent_h, latent_w),
        patch_size=patch_size,
        in_channels=latent_channels, # Must match latent_channels
        hidden_size=128, # Smaller for test
        depth=2,         # Smaller for test
        num_heads=4
    ).to(device)

    batch_size = 4
    dummy_latent_input = torch.randn(batch_size, latent_channels, latent_h, latent_w).to(device)
    dummy_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

    print(f"DiT Input latent shape: {dummy_latent_input.shape}")
    print(f"DiT Input timesteps shape: {dummy_timesteps.shape}")

    predicted_noise = dit_model(dummy_latent_input, dummy_timesteps)
    print(f"DiT Predicted noise shape: {predicted_noise.shape}") # Should match dummy_latent_input

    assert predicted_noise.shape == dummy_latent_input.shape, "Output shape mismatch!"
    print("\nDiT model test completed.")

    # Test with einops if installed
    try:
        from einops import rearrange
        # Test patchify and unpatchify logic more directly if needed
        # Patchify using einops for verification (if you were to implement patch_embed using it)
        # x_p = rearrange(dummy_latent_input, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        # print(f"Einops patched shape: {x_p.shape}") # (B, num_patches, P*P*C)
        
        # Test unpatchify logic in DiT
        # Create dummy patches
        num_patches_test = (latent_h // patch_size) * (latent_w // patch_size)
        patch_embed_dim_test = latent_channels * patch_size * patch_size
        dummy_patches_output = torch.randn(batch_size, num_patches_test, patch_embed_dim_test).to(device)
        
        unpatched_img = dit_model.unpatchify(dummy_patches_output)
        print(f"DiT Unpatchify output shape: {unpatched_img.shape}")
        assert unpatched_img.shape == (batch_size, latent_channels, latent_h, latent_w)
        print("DiT unpatchify test completed.")

    except ImportError:
        print("einops not installed, skipping some advanced patch tests.")
