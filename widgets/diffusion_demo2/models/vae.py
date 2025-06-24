import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128] # For 48x48 input -> 48/16 = 3x3 feature map

        modules = []
        in_channels = input_channels
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
        # Assuming input 48x48, after len(hidden_dims) strides of 2: 48 / (2^len(hidden_dims))
        # For default hidden_dims = [32, 64, 128, 256] (4 layers), final_spatial_dim = 48 / (2^4) = 3
        final_spatial_dim = 48 // (2**len(hidden_dims)) # Dynamically calculate based on IMAGE_SIZE if passed
        self.fc_mu = nn.Linear(hidden_dims[-1] * final_spatial_dim * final_spatial_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * final_spatial_dim * final_spatial_dim, latent_dim)

    def forward(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, output_channels=3, hidden_dims=None, image_size=48):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32] # Reverse of typical encoder
        
        self.latent_dim = latent_dim
        # Calculate initial spatial dimension for decoder based on number of transpose conv layers
        # If len(hidden_dims) = 4, then initial spatial_dim * (2^4) = image_size
        self.num_upsamples = len(hidden_dims) # Each block in hidden_dims + final layer upsamples
        initial_spatial_dim = image_size // (2**self.num_upsamples)

        self.decoder_input_fc = nn.Linear(latent_dim, hidden_dims[0] * initial_spatial_dim * initial_spatial_dim)
        self.hidden_dims = hidden_dims
        self.initial_spatial_dim = initial_spatial_dim

        modules = []
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
        x = x.view(-1, self.hidden_dims[0], self.initial_spatial_dim, self.initial_spatial_dim) 
        x = self.decoder_conv(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256, hidden_dims_enc=None, hidden_dims_dec=None, image_size=48):
        super().__init__()
        # Ensure Encoder hidden_dims are passed if not None
        self.encoder = Encoder(input_channels, latent_dim, hidden_dims=hidden_dims_enc) 
        # Pass image_size to Decoder
        self.decoder = Decoder(latent_dim, input_channels, hidden_dims=hidden_dims_dec, image_size=image_size)

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

# --- VQ-VAE Parts ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, latents): 
        latents_transposed = latents.permute(0, 2, 3, 1).contiguous() 
        latents_flat = latents_transposed.view(-1, self.embedding_dim)
        
        distances = (torch.sum(latents_flat**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(latents_flat, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=latents.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized_latents_flat = torch.matmul(encodings, self.embedding.weight)
        quantized_latents = quantized_latents_flat.view_as(latents_transposed)
        quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()
        
        e_latent_loss = F.mse_loss(quantized_latents.detach(), latents)
        q_latent_loss = F.mse_loss(quantized_latents, latents.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized_latents = latents + (quantized_latents - latents).detach() 
        
        return quantized_latents, loss, encoding_indices.view(latents_transposed.shape[:-1])


class VQVAE(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=256, num_embeddings=512, commitment_cost=0.25, 
                 hidden_dims_enc=None, hidden_dims_dec=None, image_size=48): # Added image_size
        super().__init__()
        
        if hidden_dims_enc is None: 
            hidden_dims_enc = [128, 256] 
        
        num_enc_downsamples = len(hidden_dims_enc) # +1 for the final conv to embedding_dim

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
        # This conv changes channel to embedding_dim but keeps spatial size
        modules_enc.append(nn.Conv2d(in_c, embedding_dim, kernel_size=1)) 
        self.encoder = nn.Sequential(*modules_enc)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        if hidden_dims_dec is None: # These are intermediate dimensions for the decoder part
            hidden_dims_dec = [256, 128] 

        num_dec_upsamples = len(hidden_dims_dec) + 1 # +1 for the final Tanh layer

        # The number of upsampling stages in decoder must match downsampling in encoder
        # to restore original image_size.
        # If encoder has `N` stride=2 convs, decoder needs `N` stride=2 transpose_convs.
        # Here, `len(hidden_dims_enc)` convs are stride=2.
        # `len(hidden_dims_dec)` Tconvs are stride=2, plus one final Tconv.

        if num_enc_downsamples != num_dec_upsamples:
             # Adjust hidden_dims_dec to match the required number of upsampling stages
             # This is a simple heuristic; more robust logic might be needed for arbitrary dims
            diff = num_enc_downsamples - num_dec_upsamples
            if diff > 0: # Need more upsampling stages in decoder
                # Extend hidden_dims_dec with copies of its last element or a default value
                last_dim = hidden_dims_dec[-1] if hidden_dims_dec else embedding_dim // 2
                hidden_dims_dec.extend([last_dim] * diff)
            elif diff < 0: # Need fewer upsampling stages
                hidden_dims_dec = hidden_dims_dec[:num_enc_downsamples-1] # -1 because of final layer

        modules_dec = []
        in_c_dec = embedding_dim # Decoder starts with embedding_dim channels
        current_dims = hidden_dims_dec + [input_channels] # Add output_channels for the loop structure

        for i in range(num_enc_downsamples): # Ensure same number of upsamples as downsamples
            out_c_dec = current_dims[i] if i < len(current_dims)-1 else current_dims[-1] # Target channels for this layer
            is_final_layer = (i == num_enc_downsamples -1)

            seq = [
                nn.ConvTranspose2d(in_c_dec, out_c_dec, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
            if not is_final_layer:
                seq.extend([nn.BatchNorm2d(out_c_dec), nn.LeakyReLU()])
            else: # Final layer
                seq.append(nn.Tanh())
            
            modules_dec.append(nn.Sequential(*seq))
            in_c_dec = out_c_dec
        
        self.decoder = nn.Sequential(*modules_dec)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _ = self.vq_layer(z_e)
        reconstruction = self.decoder(z_q)
        return reconstruction, vq_loss, z_e, z_q
    
    def encode(self, x):
        z_e = self.encoder(x)
        z_q, _, encoding_indices = self.vq_layer(z_e)
        return z_q, encoding_indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def sample(self, num_samples, device, codebook_indices=None):
        # Determine H_vq, W_vq from a dummy forward pass through encoder
        # Use image_size if available, else a default like 48
        # This part depends on knowing image_size at model init or passing it here.
        # Assuming image_size attribute or using a typical one.
        example_input_height = example_input_width = getattr(self, 'image_size', 48)

        dummy_input = torch.randn(1, self.encoder[0][0].in_channels, # Get in_channels from first conv layer
                                  example_input_height, example_input_width).to(device)
        z_e_dummy = self.encoder(dummy_input)
        _, _, H_vq, W_vq = z_e_dummy.shape

        if codebook_indices is None:
            codebook_indices = torch.randint(0, self.vq_layer.num_embeddings, 
                                            (num_samples, H_vq, W_vq), device=device)
        
        one_hot_indices = F.one_hot(codebook_indices.view(-1), self.vq_layer.num_embeddings).float()
        z_q_flat = torch.matmul(one_hot_indices, self.vq_layer.embedding.weight)
        z_q_reshaped = z_q_flat.view(num_samples, H_vq, W_vq, self.vq_layer.embedding_dim)
        z_q = z_q_reshaped.permute(0, 3, 1, 2)
        
        samples = self.decode(z_q)
        return samples

# --- GAN Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, hidden_dims=None, image_size=48):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512] 

        modules = []
        in_c = input_channels
        # For 48x48 input: 48 -> 24 -> 12 -> 6 -> 3
        # Number of downsampling layers:
        num_layers = 0
        current_size = image_size
        while current_size > 4 and num_layers < len(hidden_dims) : # Stop when spatial dim is small or run out of h_dims
            current_size //= 2
            num_layers +=1
        
        actual_hidden_dims = hidden_dims[:num_layers]

        for i, h_dim in enumerate(actual_hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_c, h_dim, kernel_size=4, stride=2, padding=1, bias=False),
                    # No BatchNorm on the first conv layer as per some GAN practices
                    nn.BatchNorm2d(h_dim) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_c = h_dim
        
        # Final layer to output a single logit
        # Calculate final spatial dimension after convolutions
        final_spatial_dim = image_size // (2**len(actual_hidden_dims))
        modules.append(nn.Conv2d(in_c, 1, kernel_size=final_spatial_dim, stride=1, padding=0, bias=False))
        # No sigmoid here if using BCEWithLogitsLoss

        self.discriminator = nn.Sequential(*modules)

    def forward(self, x):
        return self.discriminator(x) # Output shape: (B, 1, 1, 1)


# Test for vae.py
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    IMG_SIZE = 48 # Example image size

    # Test VAE
    print("\nTesting VAE...")
    vae_model = VAE(input_channels=3, latent_dim=128, image_size=IMG_SIZE).to(device)
    dummy_input_vae = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
    reconstruction_vae, mu, logvar, z = vae_model(dummy_input_vae)
    print(f"VAE Input shape: {dummy_input_vae.shape}")
    print(f"VAE Reconstruction shape: {reconstruction_vae.shape}")
    assert reconstruction_vae.shape == dummy_input_vae.shape, "VAE output shape mismatch"
    print(f"VAE Mu shape: {mu.shape}")
    print(f"VAE Logvar shape: {logvar.shape}")
    print(f"VAE Latent z shape: {z.shape}")
    
    samples_vae = vae_model.sample(5, device)
    print(f"VAE Generated samples shape: {samples_vae.shape}")
    assert samples_vae.shape == (5, 3, IMG_SIZE, IMG_SIZE), "VAE sample shape mismatch"


    # Test VQVAE
    print("\nTesting VQVAE...")
    vqvae_model = VQVAE(input_channels=3, embedding_dim=64, num_embeddings=128, image_size=IMG_SIZE,
                        hidden_dims_enc=[128, 64], # 2 downsamples
                        hidden_dims_dec=[64]       # expects 1 value for 2 upsamples (emb_dim -> 64 -> 3_out)
                        ).to(device)
    dummy_input_vqvae = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
    reconstruction_vqvae, vq_loss, z_e, z_q = vqvae_model(dummy_input_vqvae)
    print(f"VQVAE Input shape: {dummy_input_vqvae.shape}")
    print(f"VQVAE z_e shape (before VQ): {z_e.shape}") 
    print(f"VQVAE z_q shape (after VQ): {z_q.shape}")   
    print(f"VQVAE Reconstruction shape: {reconstruction_vqvae.shape}")
    assert reconstruction_vqvae.shape == dummy_input_vqvae.shape, "VQVAE output shape mismatch"
    print(f"VQVAE VQ Loss: {vq_loss.item()}")

    z_q_encoded, indices = vqvae_model.encode(dummy_input_vqvae)
    print(f"VQVAE Encoded z_q shape: {z_q_encoded.shape}")
    print(f"VQVAE Encoded indices shape: {indices.shape}") 

    samples_vqvae = vqvae_model.sample(5, device)
    print(f"VQVAE Generated samples shape: {samples_vqvae.shape}")
    assert samples_vqvae.shape == (5, 3, IMG_SIZE, IMG_SIZE), "VQVAE sample shape mismatch"

    # Test Discriminator
    print("\nTesting Discriminator...")
    discriminator_model = Discriminator(input_channels=3, image_size=IMG_SIZE).to(device)
    dummy_input_disc = torch.randn(4, 3, IMG_SIZE, IMG_SIZE).to(device)
    output_disc = discriminator_model(dummy_input_disc)
    print(f"Discriminator Input shape: {dummy_input_disc.shape}")
    print(f"Discriminator Output shape: {output_disc.shape}") # Should be (B, 1, 1, 1)
    assert output_disc.shape == (4, 1, 1, 1), "Discriminator output shape mismatch"

    print("\nModel tests completed.")
