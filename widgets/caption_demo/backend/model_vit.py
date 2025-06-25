import torch
import torch.nn as nn
import torch.nn.functional as F
import timm # For ViT encoder
import math
from backend.data_utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN # For token IDs in generation

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ViTDecoder(nn.Module):
    """Simplified Transformer Decoder from scratch"""
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_feedforward_dim, dropout=0.1, max_seq_len=50):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout, max_len=max_seq_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads,
            dim_feedforward=hidden_feedforward_dim,
            dropout=dropout,
            batch_first=True # Important!
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask # .to(device) # device is already set

    def forward(self, encoder_features, tgt_captions):
        """
        encoder_features: (batch_size, num_patches, embed_size) - from ViT encoder
        tgt_captions: (batch_size, seq_len) - target caption tokens (shifted right)
        """
        device = tgt_captions.device
        tgt_emb = self.embedding(tgt_captions) * math.sqrt(self.embed_size) # (batch, seq_len, embed_size)
        tgt_emb = self.pos_encoder(tgt_emb) # Add positional encoding

        tgt_mask = self.generate_square_subsequent_mask(tgt_captions.size(1), device)
        # tgt_key_padding_mask can be generated from PAD tokens in tgt_captions

        # memory_key_padding_mask can be used if encoder_features have padding (not typical for ViT CLS/patch tokens directly)

        output = self.transformer_decoder(
            tgt=tgt_emb,                # (batch, tgt_seq_len, embed_size)
            memory=encoder_features,    # (batch, src_seq_len/num_patches, embed_size)
            tgt_mask=tgt_mask           # (tgt_seq_len, tgt_seq_len)
            # tgt_key_padding_mask=tgt_key_padding_mask,
            # memory_key_padding_mask=memory_key_padding_mask
        ) # (batch, seq_len, embed_size)
        
        logits = self.fc_out(output) # (batch, seq_len, vocab_size)
        return logits

class ViTImageCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_size=768, num_heads=8, num_decoder_layers=6, 
                 decoder_ff_dim=2048, dropout=0.1, vit_model_name='vit_base_patch16_224', 
                 max_seq_len=50, tokenizer=None):
        super().__init__()
        self.vit_model_name = vit_model_name
        self.embed_size = embed_size # Decoder's embed_size

        # Load pre-trained ViT encoder
        self.encoder = timm.create_model(vit_model_name, pretrained=True, num_classes=0) # num_classes=0 for feature extraction
        
        # Modify ViT to output patch features
        if hasattr(self.encoder, 'global_pool') and self.encoder.global_pool is not None:
             # Check if global_pool is already an Identity or similar, or a module that can be replaced
            if not isinstance(self.encoder.global_pool, nn.Identity):
                 self.encoder.global_pool = nn.Identity() # Disable global pooling
                 print(f"Disabled global_pool for {vit_model_name} to get patch features.")
        if hasattr(self.encoder, 'head') and not isinstance(self.encoder.head, nn.Identity):
            self.encoder.head = nn.Identity()        # Disable classification head
            print(f"Disabled head for {vit_model_name} to get patch features.")


        # Get ViT's expected input image size
        if hasattr(self.encoder, 'patch_embed') and hasattr(self.encoder.patch_embed, 'img_size'):
            self.image_size = self.encoder.patch_embed.img_size # (H, W) tuple
        elif hasattr(self.encoder, 'img_size'): # Fallback for some models, though less common for ViTs
             self.image_size = self.encoder.img_size
        else:
            # As a last resort, try to infer from model name or use a default
            # For 'vit_tiny_patch16_224', it's (224, 224)
            # This part might need adjustment if using diverse timm models
            if '224' in vit_model_name:
                self.image_size = (224, 224)
                print(f"Warning: Could not directly determine img_size. Assuming {self.image_size} based on model name.")
            else:
                raise AttributeError(f"ViT model {vit_model_name} does not have a recognizable img_size attribute. Please check model structure or specify image_size.")
        
        dummy_input = torch.randn(1, 3, self.image_size[0], self.image_size[1])
        
        try:
            test_features = self.encoder.forward_features(dummy_input)
        except AttributeError: 
            print(f"Warning: ViT model {vit_model_name} may not have a standard 'forward_features' method. Attempting to use default forward().")
            test_features = self.encoder(dummy_input) 
        
        if test_features.ndim == 2: 
             print(f"Warning: ViT model {vit_model_name} output is (B, D). Patch features preferred. Model may not work as expected with this feature shape for attention.")
        
        vit_feature_dim = test_features.shape[-1]
        
        # Projection layer if ViT's feature dim doesn't match decoder's expected embed_size
        self.input_proj = nn.Linear(vit_feature_dim, embed_size) if vit_feature_dim != embed_size else nn.Identity()

        self.decoder = ViTDecoder(vocab_size, embed_size, num_heads, num_decoder_layers, 
                                  decoder_ff_dim, dropout, max_seq_len)
        self.tokenizer = tokenizer
        # self.image_size is already set above

        # Freeze encoder by default
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"ViT encoder ({vit_model_name}) parameters frozen.")

    def forward(self, images, captions_input):
        current_img_size = (images.shape[2], images.shape[3])
        if current_img_size != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode='bilinear', align_corners=False)

        try:
            encoder_output = self.encoder.forward_features(images) 
        except AttributeError:
            encoder_output = self.encoder(images) # Fallback if forward_features isn't present
        
        encoder_features = self.input_proj(encoder_output) 
        
        logits = self.decoder(encoder_features, captions_input)
        return logits

    def generate_caption(self, image_tensor, max_len=50, device='cpu'):
        self.eval()
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set for generation.")

        sos_idx = self.tokenizer.token_to_id(SOS_TOKEN)
        eos_idx = self.tokenizer.token_to_id(EOS_TOKEN)
        # pad_idx = self.tokenizer.token_to_id(PAD_TOKEN) # Not used in greedy generation output

        if image_tensor.ndim == 3: # If single image (C, H, W)
            image_tensor = image_tensor.unsqueeze(0) # Add batch dimension

        image_tensor = image_tensor.to(device)

        if image_tensor.shape[2:] != self.image_size:
            image_tensor = F.interpolate(image_tensor, size=self.image_size, mode='bilinear', align_corners=False)

        with torch.no_grad():
            try:
                encoder_output = self.encoder.forward_features(image_tensor)
            except AttributeError:
                encoder_output = self.encoder(image_tensor)
            encoder_features = self.input_proj(encoder_output) # (1, num_patches, embed_size)

            tgt_caption_ids = torch.LongTensor([[sos_idx]]).to(device) # (1, 1)

            for _ in range(max_len -1): # Max_len includes SOS
                output_logits = self.decoder(encoder_features, tgt_caption_ids) # (1, current_len, vocab_size)
                
                last_token_logits = output_logits[:, -1, :] # (1, vocab_size)
                predicted_idx = last_token_logits.argmax(1) # (1)
                
                tgt_caption_ids = torch.cat([tgt_caption_ids, predicted_idx.unsqueeze(1)], dim=1)

                if predicted_idx.item() == eos_idx:
                    break
        
        generated_ids = tgt_caption_ids.squeeze(0).tolist() # squeeze(0) to remove batch dim
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# Self-test
if __name__ == "__main__":
    print("Testing model_vit.py...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy params
    vocab_s = 1000
    embed_s = 256 
    
    vit_model_name_test = 'vit_tiny_patch16_224' 
    
    num_h = 4
    num_dec_layers = 3
    dec_ff_dim = embed_s * 2 
    max_s_len = 20
    batch_s = 2
    
    # Mock tokenizer for generation test
    class MockTokenizer:
        def token_to_id(self, token):
            if token == SOS_TOKEN: return 1
            if token == EOS_TOKEN: return 2
            if token == PAD_TOKEN: return 0
            return 3 # unk
        def decode(self, ids, skip_special_tokens=True):
            map_dict = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: "word"}
            tokens = []
            # Corrected logic for skip_special_tokens to actually skip them
            for _id_val in ids:
                is_special = _id_val in [self.token_to_id(PAD_TOKEN), self.token_to_id(SOS_TOKEN), self.token_to_id(EOS_TOKEN)]
                if skip_special_tokens and is_special:
                    continue
                tokens.append(map_dict.get(_id_val, "unknown"))
            return " ".join(tokens)
        def get_vocab_size(self): return vocab_s

    try:
        model = ViTImageCaptionModel(
            vocab_size=vocab_s, 
            embed_size=embed_s, 
            num_heads=num_h, 
            num_decoder_layers=num_dec_layers,
            decoder_ff_dim=dec_ff_dim,
            vit_model_name=vit_model_name_test,
            max_seq_len=max_s_len,
            tokenizer=MockTokenizer()
        ).to(device)

        dummy_images = torch.randn(batch_s, 3, 48, 48).to(device) 
        dummy_captions_in = torch.randint(0, vocab_s, (batch_s, max_s_len)).to(device)

        output_logits = model(dummy_images, dummy_captions_in)
        print(f"ViT Model output shape: {output_logits.shape}") 
        assert output_logits.shape == (batch_s, max_s_len, vocab_s)
        
        single_image = torch.randn(1, 3, 48, 48) # No .to(device) here, handled in generate_caption
        generated_caption = model.generate_caption(single_image, max_len=10, device=device) # Pass device
        print(f"Generated ViT caption (dummy): '{generated_caption}'")
        assert isinstance(generated_caption, str)

        print("model_vit.py tests passed!")

    except Exception as e:
        print(f"Error during model_vit.py test: {e}")
        import traceback
        traceback.print_exc()
        print("This might be due to ViT model download/compatibility or architectural assumptions.")
        print("Ensure 'timm' is installed and an internet connection is available for model download.")
