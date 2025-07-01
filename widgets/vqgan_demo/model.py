# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VectorQuantizer(nn.Module):
    """
    Improved VectorQuantizer module with separate commitment and codebook losses.
    This module takes a float tensor of shape [B, C, H, W] and maps it to a
    discrete set of codes from a learned codebook.
    """
    def __init__(self, n_embed, embed_dim, commitment_cost):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        # Initialize embeddings similar to the official VQGAN implementation
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self, z):
        """
        Forward pass.
        Args:
            z (torch.Tensor): The continuous latent vector from the encoder.
                              Shape: [B, C, H, W]
        Returns:
            torch.Tensor: Quantized latent vector. Shape: [B, C, H, W]
            torch.Tensor: Codebook loss (scalar).
            torch.Tensor: Commitment loss (scalar).
            torch.Tensor: The indices of the closest embeddings. Shape: [B, H, W]
        """
        # z shape: [B, C, H, W]
        B, C, H, W = z.shape
        
        # Permute and reshape z from [B, C, H, W] to [B*H*W, C]
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        z_reshaped = z_permuted.view(-1, self.embed_dim)

        # Calculate distances between each z vector and each embedding vector
        # (a-b)^2 = a^2 + b^2 - 2ab
        distances = (torch.sum(z_reshaped**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z_reshaped, self.embedding.weight.t()))

        # Find the closest embedding for each vector
        min_encoding_indices = torch.argmin(distances, dim=1)
        
        # Get the quantized vectors from the embedding table and reshape back to [B, H, W, C]
        z_q_permuted = self.embedding(min_encoding_indices).view(z_permuted.shape)

        # Calculate losses
        # Both z_permuted and z_q_permuted are in [B, H, W, C] format
        # Codebook loss: moves the embedding vectors towards the encoder outputs
        codebook_loss = F.mse_loss(z_q_permuted, z_permuted.detach())
        # Commitment loss: moves the encoder outputs towards the embedding vectors
        commitment_loss = self.commitment_cost * F.mse_loss(z_permuted, z_q_permuted.detach())

        # Straight-through estimator: copy gradients from z_q to z
        z_q_permuted = z_permuted + (z_q_permuted - z_permuted).detach()
        
        # Permute z_q back to the standard [B, C, H, W] format for the decoder
        z_q = z_q_permuted.permute(0, 3, 1, 2).contiguous()
        
        # Reshape indices for transformer input
        min_encoding_indices = min_encoding_indices.view(B, H, W)

        return z_q, codebook_loss, commitment_loss, min_encoding_indices


class ResnetBlock(nn.Module):
    """Basic residual block for CNNs."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
            
        return x + h

class AttnBlock(nn.Module):
    """Self-attention block."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k) * (c ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        v = v.permute(0, 2, 1)  # b,hw,c
        h_ = torch.bmm(w_, v)
        h_ = h_.permute(0, 2, 1).reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_

class Encoder(nn.Module):
    """The Encoder part of the VQGAN. Downsamples the image."""
    def __init__(self, in_channels, ch, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        
        self.down = nn.ModuleList()
        for i_level in range(len(ch_mult)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            
            down = nn.Module()
            down.block = block
            if curr_res in attn_resolutions:
                attn.append(AttnBlock(block_out))
            down.attn = attn
            
            if i_level != len(ch_mult) - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=4, stride=2, padding=1)
                curr_res //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_out)
        self.mid.attn_1 = AttnBlock(block_out)
        self.mid.block_2 = ResnetBlock(block_out)

        self.norm_out = nn.GroupNorm(32, block_out)
        self.conv_out = nn.Conv2d(block_out, 256, kernel_size=3, stride=1, padding=1) # embed_dim is 256

    def forward(self, x):
        x = self.conv_in(x)
        for down in self.down:
            for block in down.block:
                x = block(x)
            for attn in down.attn:
                x = attn(x)
            if hasattr(down, 'downsample'):
                x = down.downsample(x)
        
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    """The Decoder part of the VQGAN. Upsamples to reconstruct the image."""
    def __init__(self, out_channels, ch, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(256, block_in, kernel_size=3, padding=1) # embed_dim is 256

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in)

        self.up = nn.ModuleList()
        # ---- START OF FIX ----
        # Loop from the deepest level to the shallowest
        for i_level in reversed(range(len(ch_mult))):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            up = nn.Module()

            # Define the upsampling and convolution layers BEFORE `block_in` is updated.
            # The convolution takes `block_in` channels (from the deeper level) and
            # outputs `block_in` channels, which is the expected input for the first ResBlock.
            if i_level != 0:
                up.upsample = nn.Upsample(scale_factor=2, mode="nearest")
                up.conv = nn.Conv2d(block_in, block_in, kernel_size=3, padding=1)
                resolution *= 2

            # Define ResNet blocks. The first block transitions from the previous level's
            # channel count (`block_in`) to the current level's (`block_out`).
            # Subsequent blocks operate on `block_out` channels.
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out # Update `block_in` for the next ResBlock in this level
            
            up.block = block

            # Attention is applied after the ResBlocks, so it uses the updated channel count.
            if resolution in attn_resolutions:
                attn.append(AttnBlock(block_in)) # Here, block_in == block_out
            up.attn = attn

            # Use append to build the list in the correct processing order (deepest to shallowest).
            self.up.append(up)
        # ---- END OF FIX ----

        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        z = self.conv_in(z)
        z = self.mid.block_1(z)
        z = self.mid.attn_1(z)
        z = self.mid.block_2(z)

        for up in self.up:
            if hasattr(up, 'upsample'):
                z = up.upsample(z)
                z = up.conv(z)
            for block in up.block:
                z = block(z)
            for attn in up.attn:
                z = attn(z)
        
        z = self.norm_out(z)
        z = F.silu(z)
        z = self.conv_out(z)
        return z


class VQGAN(nn.Module):
    """The complete VQGAN model for stage 1 training."""
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            in_channels=config['in_channels'],
            ch=config['ch'],
            ch_mult=config['ch_mult'],
            num_res_blocks=config['num_res_blocks'],
            resolution=config['resolution'],
            attn_resolutions=config['attn_resolutions']
        )
        self.quantizer = VectorQuantizer(
            n_embed=config['n_embed'],
            embed_dim=config['embed_dim'],
            commitment_cost=config['commitment_cost']
        )
        self.decoder = Decoder(
            out_channels=config['out_channels'],
            ch=config['ch'],
            ch_mult=config['ch_mult'],
            num_res_blocks=config['num_res_blocks'],
            resolution=config['resolution'],
            attn_resolutions=config['attn_resolutions']
        )
        self.quant_conv = nn.Conv2d(config['embed_dim'], config['embed_dim'], 1)
        self.post_quant_conv = nn.Conv2d(config['embed_dim'], config['embed_dim'], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, codebook_loss, commitment_loss, indices = self.quantizer(h)
        return quant, codebook_loss, commitment_loss, indices

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def decode_from_indices(self, indices):
        """Decode from a batch of code indices."""
        # Get embeddings from indices
        z_q = self.quantizer.embedding(indices) # Shape: B, H, W, C
        z_q = z_q.permute(0, 3, 1, 2) # Shape: B, C, H, W
        return self.decode(z_q)

    def forward(self, x):
        quant, codebook_loss, commitment_loss, _ = self.encode(x)
        x_rec = self.decode(quant)
        return x_rec, codebook_loss, commitment_loss


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix."""
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class VQGANTransformer(nn.Module):
    """
    The Transformer model for stage 2 training.
    It learns to predict the next code index in a sequence.
    """
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config['vocab_size'] # n_embed
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.block_size = config['block_size'] # sequence length (e.g., 8*8)

        # Token and position embeddings
        self.tok_emb = nn.Embedding(self.vocab_size + 1, self.n_embd) # +1 for start token
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))

        # Transformer blocks
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=self.n_head,
            dim_feedforward=4 * self.n_embd,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=self.n_layer)

        # Output head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

    def generate_square_subsequent_mask(self, sz, device):
        """Generates a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, idx):
        # idx is shape (B, T)
        B, T = idx.shape
        device = idx.device
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        # Token embeddings
        token_embeddings = self.tok_emb(idx) # (B, T, n_embd)

        # Position embeddings
        pos_embeddings = self.pos_emb[:, :T, :] # (1, T, n_embd)

        x = token_embeddings + pos_embeddings

        # --- 关键补充：创建并应用因果掩码 ---
        causal_mask = self.generate_square_subsequent_mask(T, device)
        x = self.transformer(x, mask=causal_mask)
        # ------------------------------------

        x = self.ln_f(x)
        logits = self.head(x) # (B, T, vocab_size)

        return logits


    @torch.no_grad()
    def sample(self, n_samples, seq_len, start_token, device, temperature=1.0, top_k=None):
        """Autoregressively sample new sequences of codes."""
        # Start with a batch of start_token
        # The start token has an index of self.vocab_size
        indices = torch.full((n_samples, 1), start_token, dtype=torch.long, device=device)
        
        for _ in range(seq_len):
            # Crop to block size if sequence gets too long
            idx_cond = indices if indices.size(1) <= self.block_size else indices[:, -self.block_size:]
            
            # Forward pass
            logits = self(idx_cond)
            # Pluck the logits at the final step
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            indices = torch.cat((indices, idx_next), dim=1)
            
        # Return generated indices, removing the start token
        return indices[:, 1:]

# --- Unit Tests ---
if __name__ == '__main__':
    def test_vqgan_stage1():
        print("--- Running VQGAN Stage 1 Unit Test ---")
        config = {
            'in_channels': 3,
            'out_channels': 3,
            'ch': 128,
            'ch_mult': [1, 1, 2, 2, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'resolution': 64,
            'embed_dim': 256,
            'n_embed': 1024,
            'commitment_cost': 0.25
        }
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Test VQGAN
        model = VQGAN(config).to(device)
        dummy_input = torch.randn(4, 3, 64, 64).to(device)
        
        print(f"Input shape: {dummy_input.shape}")
        
        x_rec, codebook_loss, commit_loss = model(dummy_input)
        
        assert x_rec.shape == dummy_input.shape, f"Shape mismatch: {x_rec.shape} vs {dummy_input.shape}"
        assert codebook_loss.ndim == 0, "Codebook loss should be a scalar"
        assert commit_loss.ndim == 0, "Commitment loss should be a scalar"
        print(f"Reconstruction shape: {x_rec.shape}")
        print(f"Codebook Loss: {codebook_loss.item():.4f}, Commitment Loss: {commit_loss.item():.4f}")
        print("VQGAN forward pass successful.")
        
        # Test encode/decode
        _, _, _, indices = model.encode(dummy_input)
        # 64 / 2**(len(ch_mult)-1) = 64 / 2**4 = 64/16 = 4
        assert indices.shape == (4, 4, 4), f"Indices shape mismatch: {indices.shape}"
        print(f"Encoded indices shape: {indices.shape}")
        
        x_rec_from_indices = model.decode_from_indices(indices)
        assert x_rec_from_indices.shape == dummy_input.shape
        print("Decode from indices successful.")

        # Test Discriminator
        discriminator = NLayerDiscriminator(input_nc=3, n_layers=3).to(device)
        disc_output_real = discriminator(dummy_input)
        disc_output_fake = discriminator(x_rec.detach())
        
        # Expected output shape: for 64x64 input and 3 layers, downsamples 3 times -> 64/8=8 -> 5x5 output patch grid
        # The receptive field calculation is a bit complex, but we can just check the output shape
        # The output of discriminator is Bx1xHxW
        assert disc_output_real.shape[0] == 4 and disc_output_real.shape[1] == 1
        print(f"Discriminator output shape: {disc_output_real.shape}")
        print("Discriminator forward pass successful.")
        print("--- VQGAN Stage 1 Unit Test PASSED ---\n")

    def test_transformer_stage2():
        print("--- Running VQGAN Transformer Stage 2 Unit Test ---")
        config = {
            'vocab_size': 1024,
            'block_size': 8*8, # 64
            'n_layer': 8,
            'n_head': 8,
            'n_embd': 512,
        }
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        model = VQGANTransformer(config).to(device)
        
        # Test forward pass
        # The latent map from stage 1 is 4x4=16, but we test with the configured block_size
        test_seq_len = 4*4 
        dummy_indices = torch.randint(0, config['vocab_size'], (4, test_seq_len)).to(device)
        print(f"Input indices shape: {dummy_indices.shape}")
        
        logits = model(dummy_indices)
        expected_shape = (4, test_seq_len, config['vocab_size'])
        assert logits.shape == expected_shape, f"Logits shape mismatch: {logits.shape} vs {expected_shape}"
        print(f"Output logits shape: {logits.shape}")
        print("Transformer forward pass successful.")

        # Test sampling
        n_samples = 2
        seq_len = test_seq_len
        start_token = config['vocab_size'] # Special start token
        
        generated_indices = model.sample(n_samples, seq_len, start_token, device=device, top_k=50)
        expected_shape = (n_samples, seq_len)
        assert generated_indices.shape == expected_shape, f"Sampled indices shape mismatch: {generated_indices.shape} vs {expected_shape}"
        print(f"Sampled indices shape: {generated_indices.shape}")
        print("Transformer sampling successful.")
        print("--- VQGAN Transformer Stage 2 Unit Test PASSED ---")

    test_vqgan_stage1()
    test_transformer_stage2()
