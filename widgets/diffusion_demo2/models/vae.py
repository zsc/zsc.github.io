import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        modules = []
        in_channels = input_channels
        # Example for 48x48 input
        # 48 -> 24 -> 12 -> 6
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        self.encoder_conv = nn.Sequential(*modules)
        # Assuming input 48x48, after 4 strides of 2: 48 / (2^4) = 48/16 = 3
        # So feature map size is 3x3
        self.fc_mu = nn.Linear(hidden_dims[-1] * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 3 * 3, latent_dim)

    def forward(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, output_channels=3, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        self.latent_dim = latent_dim
        # Assuming final feature map size before output conv is 3x3
        self.decoder_input_fc = nn.Linear(latent_dim, hidden_dims[0] * 3 * 3)
        self.hidden_dims = hidden_dims

        modules = []
        # 3 -> 6 -> 12 -> 24 -> 48
        in_channels = hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, hidden_dims[i+1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )
            in_channels = hidden_dims[i+1]
        
        # Final layer to get back to image size and channels
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], output_channels,
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Tanh()) # Output in [-1, 1] range
        )
        self.decoder_conv = nn.Sequential(*modules)

    def forward(self, z):
        x = self.decoder_input_fc(z)
        x = x.view(-1, self.hidden_dims[0], 3, 3) # Reshape to feature map
        x = self.decoder_conv(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256, hidden_dims_enc=None, hidden_dims_dec=None):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim, hidden_dims_enc)
        self.decoder = Decoder(latent_dim, input_channels, hidden_dims_dec)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar, z
    
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features).to(device)
        samples = self.decoder(z)
        return samples

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

# --- VQ-VAE Parts (Simplified) ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, latents): # latents: (B, C, H, W)
        latents_transposed = latents.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        latents_flat = latents_transposed.view(-1, self.embedding_dim) # (B*H*W, C)
        
        # Calculate distances
        distances = (torch.sum(latents_flat**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(latents_flat, self.embedding.weight.t()))
            
        # Find closest encodings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (B*H*W, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=latents.device)
        encodings.scatter_(1, encoding_indices, 1) # One-hot
        
        # Quantize and unflatten
        quantized_latents_flat = torch.matmul(encodings, self.embedding.weight) # (B*H*W, C)
        quantized_latents = quantized_latents_flat.view_as(latents_transposed) # (B, H, W, C)
        quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        
        # Losses
        e_latent_loss = F.mse_loss(quantized_latents.detach(), latents)
        q_latent_loss = F.mse_loss(quantized_latents, latents.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized_latents = latents + (quantized_latents - latents).detach() 
        
        return quantized_latents, loss, encoding_indices.view(latents_transposed.shape[:-1])


class VQVAE(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=256, num_embeddings=512, commitment_cost=0.25, 
                 hidden_dims_enc=None, hidden_dims_dec=None, data_shape=(48,48)):
        super().__init__()
        # Simplified VQVAE: Encoder outputs features that match embedding_dim directly
        # Example for 48x48 -> 12x12 features for VQ
        # Adjust encoder/decoder to output/input appropriate feature map size
        
        # For VQVAE, encoder typically doesn't output mu/logvar, but directly features to be quantized
        # Let's make a simplified encoder for VQVAE
        if hidden_dims_enc is None: # e.g. 48 -> 24 -> 12
            hidden_dims_enc = [128, 256] 
        
        modules_enc = []
        in_c = input_channels
        for h_dim in hidden_dims_enc:
            modules_enc.append(
                nn.Sequential(
                    nn.Conv2d(in_c, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim), nn.LeakyReLU()
                )
            )
            in_c = h_dim
        modules_enc.append(nn.Conv2d(in_c, embedding_dim, kernel_size=1)) # To match embedding_dim
        self.encoder = nn.Sequential(*modules_enc)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder needs to take quantized latents. Input channels for decoder = embedding_dim
        # And its hidden_dims might need adjustment based on the encoder's output feature map size
        # If encoder outputs 12x12 features, decoder starts from 12x12
        # Example: 12 -> 24 -> 48
        if hidden_dims_dec is None:
            hidden_dims_dec = [256, 128] # Start with embedding_dim as first layer input

        modules_dec = []
        in_c_dec = embedding_dim
        for h_dim in hidden_dims_dec:
            modules_dec.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_c_dec, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(h_dim), nn.LeakyReLU()
                )
            )
            in_c_dec = h_dim
        modules_dec.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_c_dec, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*modules_dec)

    def forward(self, x):
        z_e = self.encoder(x) # (B, embedding_dim, H', W')
        z_q, vq_loss, _ = self.vq_layer(z_e)
        reconstruction = self.decoder(z_q)
        return reconstruction, vq_loss, z_e, z_q
    
    def encode(self, x): # Returns quantized latents and indices
        z_e = self.encoder(x)
        z_q, _, encoding_indices = self.vq_layer(z_e)
        return z_q, encoding_indices

    def decode(self, z_q): # Takes quantized latents
        return self.decoder(z_q)

    def sample(self, num_samples, device, codebook_indices=None):
        # Sampling from VQVAE is more complex if you want diverse good samples.
        # Simplest is to pick random codebook entries and arrange them.
        # Or train a prior (e.g., PixelCNN) over codebook_indices.
        # For this demo, let's just decode random codes.
        # Assume encoder output H', W' are known (e.g. 12x12 from a 48x48 input)
        # Latent feature map H', W' from encoder: e.g. 12x12 for 48x48 with 2 downsamples
        
        # Get one of the encoder modules to find out the output shape
        dummy_input = torch.randn(1, 3, 48, 48).to(device) # Example input H,W
        z_e_dummy = self.encoder(dummy_input)
        _, C_vq, H_vq, W_vq = z_e_dummy.shape

        if codebook_indices is None:
            # Generate random codebook indices
            codebook_indices = torch.randint(0, self.vq_layer.num_embeddings, 
                                            (num_samples, H_vq, W_vq), device=device)
        
        # Convert indices to one-hot and then to embeddings
        # (num_samples * H_vq * W_vq, num_embeddings)
        one_hot_indices = F.one_hot(codebook_indices.view(-1), self.vq_layer.num_embeddings).float()
        # (num_samples * H_vq * W_vq, embedding_dim)
        z_q_flat = torch.matmul(one_hot_indices, self.vq_layer.embedding.weight)
        # (num_samples, H_vq, W_vq, embedding_dim)
        z_q_reshaped = z_q_flat.view(num_samples, H_vq, W_vq, self.vq_layer.embedding_dim)
        # (num_samples, embedding_dim, H_vq, W_vq)
        z_q = z_q_reshaped.permute(0, 3, 1, 2)
        
        samples = self.decode(z_q)
        return samples


# Test for vae.py
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test VAE
    print("\nTesting VAE...")
    vae_model = VAE(input_channels=3, latent_dim=128).to(device)
    dummy_input_vae = torch.randn(4, 3, 48, 48).to(device) # Batch size 4
    reconstruction_vae, mu, logvar, z = vae_model(dummy_input_vae)
    print(f"VAE Input shape: {dummy_input_vae.shape}")
    print(f"VAE Reconstruction shape: {reconstruction_vae.shape}")
    print(f"VAE Mu shape: {mu.shape}")
    print(f"VAE Logvar shape: {logvar.shape}")
    print(f"VAE Latent z shape: {z.shape}")
    
    samples_vae = vae_model.sample(5, device)
    print(f"VAE Generated samples shape: {samples_vae.shape}")

    # Test VQVAE
    print("\nTesting VQVAE...")
    # For VQVAE, ensure encoder output matches embedding_dim
    # Encoder output for 48x48 (2 downsamples of stride 2) -> 12x12 feature map
    # Decoder input will be (embedding_dim, 12, 12)
    vqvae_model = VQVAE(input_channels=3, embedding_dim=64, num_embeddings=128, 
                        data_shape=(48,48)).to(device)
    dummy_input_vqvae = torch.randn(4, 3, 48, 48).to(device)
    reconstruction_vqvae, vq_loss, z_e, z_q = vqvae_model(dummy_input_vqvae)
    print(f"VQVAE Input shape: {dummy_input_vqvae.shape}")
    print(f"VQVAE z_e shape (before VQ): {z_e.shape}") # Should be (B, embedding_dim, H', W')
    print(f"VQVAE z_q shape (after VQ): {z_q.shape}")   # Should be same as z_e
    print(f"VQVAE Reconstruction shape: {reconstruction_vqvae.shape}")
    print(f"VQVAE VQ Loss: {vq_loss.item()}")

    z_q_encoded, indices = vqvae_model.encode(dummy_input_vqvae)
    print(f"VQVAE Encoded z_q shape: {z_q_encoded.shape}")
    print(f"VQVAE Encoded indices shape: {indices.shape}") # Should be (B, H', W')

    samples_vqvae = vqvae_model.sample(5, device)
    print(f"VQVAE Generated samples shape: {samples_vqvae.shape}")

    print("\nModel tests completed.")
