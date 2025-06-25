import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, AutoTokenizer # Using AutoTokenizer for GPT2 later
import timm # For ViT encoder
from backend.data_utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN # For token IDs in generation

# Note: The actual GPT2 tokenizer will be used for text.
# The custom BPE tokenizer is for the overall caption vocabulary.
# There's a mismatch here if we directly feed BPE tokens to GPT2.
# A more robust MLLM would:
# 1. Use the LLM's own tokenizer for the text part.
# 2. Map image caption BPE vocabulary to LLM's vocabulary if training end-to-end.
# For this setup (frozen LLM, only adapter trained), the BPE output needs to be fed to the LLM.
# This implies the BPE vocab should ideally be compatible or a projection from BPE vocab to LLM vocab is needed.
# Simpler for demo: Assume BPE tokenizer is used, and its vocab size matches what the LLM's final layer expects.
# This means the LLM's token embedding and output LM head are NOT used directly.
# Instead, the BPE tokens are embedded, then processed by GPT2 body, then an LM head for BPE vocab.
# OR, a more common setup:
#   - Vision features are projected and prepended to text embeddings *from the LLM's own tokenizer*.
#   - The LLM then predicts tokens from its *own vocabulary*.
#   - Output captions are then tokenized by BPE for loss calculation IF BPE is the target vocab.
# This gets complex.
# For "LLM part freeze", it usually means its weights including embeddings and LM head are frozen.
# Let's simplify:
#   - We use BPE tokenizer for our captions.
#   - The MLLM will have an embedding layer for BPE tokens.
#   - Vision features are projected.
#   - GPT2 body processes concatenated vision_proj + text_bpe_embeddings.
#   - A final Linear layer maps GPT2 hidden states to BPE vocab_size.
# This means GPT2's own token embeddings and LM head are bypassed. Only its transformer blocks are used.

class MLPAdapter(nn.Module):
    """Simple MLP to project vision features."""
    def __init__(self, vision_feature_dim, llm_hidden_size, num_vision_tokens_output=16): # num_vision_tokens can be fixed or dynamic
        super().__init__()
        self.num_vision_tokens_output = num_vision_tokens_output # How many "virtual tokens" represent the image
        # This adapter could project each patch, or project the global feature num_vision_tokens_output times.
        # Let's assume vision_feature_dim is from a global image feature (e.g., CLS token or GAP)
        # And we want to "expand" it into a sequence of tokens for the LLM.
        # If vision_features are already a sequence (e.g. patch embeddings), then this can be a per-patch projection.
        
        # Scenario 1: vision_feature_dim is a single vector (e.g., [B, D_vision])
        # self.projection = nn.Linear(vision_feature_dim, num_vision_tokens_output * llm_hidden_size)
        # Then reshape to [B, num_vision_tokens_output, llm_hidden_size]

        # Scenario 2: vision_feature_dim is from patch embeddings (e.g. [B, num_patches, D_vision])
        # And we want to project each patch to llm_hidden_size, potentially subsampling/pooling patches
        # to get num_vision_tokens_output.
        # For simplicity, let's assume it's a linear projection per patch, and num_vision_tokens_output
        # is determined by the vision encoder (e.g., num_patches).
        self.projection = nn.Linear(vision_feature_dim, llm_hidden_size)
        # If num_vision_tokens_output is different from num_patches, more complex logic is needed (pooling, attention, etc.)

    def forward(self, vision_features):
        # Assuming vision_features are [Batch, NumPatches, VisionDim]
        projected_vision_features = self.projection(vision_features) # [Batch, NumPatches, LLMHiddenSize]
        return projected_vision_features # This will be our sequence of vision tokens

class MLLM(nn.Module):
    def __init__(self,
                 bpe_vocab_size: int,         # Our custom BPE tokenizer's vocab size
                 llm_embed_dim: int,          # Embedding dim for BPE tokens, should match LLM hidden size for easy concat
                 llm_hidden_size: int,        # GPT2's hidden size (e.g., 768 for gpt2-124M)
                 gpt2_model_name: str = "gpt2", # e.g., "gpt2" (124M)
                 vision_encoder_name: str = "vit_base_patch16_224",
                 # num_vision_tokens: int = 16, # How many tokens to represent the image
                 max_seq_len: int = 64, # Max combined vision + text tokens
                 tokenizer=None # Our BPE tokenizer
                ):
        super().__init__()
        self.bpe_vocab_size = bpe_vocab_size
        self.llm_embed_dim = llm_embed_dim # For our BPE tokens
        self.llm_hidden_size = llm_hidden_size # GPT2's internal dim, should match llm_embed_dim
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer # Our BPE tokenizer

        if llm_embed_dim != llm_hidden_size:
            print(f"Warning: MLLM llm_embed_dim ({llm_embed_dim}) for BPE tokens should ideally match " \
                  f"LLM hidden size ({llm_hidden_size}) for direct feature concatenation. Adjusting llm_embed_dim.")
            self.llm_embed_dim = llm_hidden_size


        # Vision Encoder (e.g., ViT)
        self.vision_encoder = timm.create_model(vision_encoder_name, pretrained=True, num_classes=0)
        if hasattr(self.vision_encoder, 'global_pool'): # Get patch features
            self.vision_encoder.global_pool = nn.Identity()
            self.vision_encoder.head = nn.Identity()
        
        dummy_img_size = self.vision_encoder.img_size if hasattr(self.vision_encoder, 'img_size') else (224,224)
        dummy_input_vision = torch.randn(1, 3, dummy_img_size[0], dummy_img_size[1])
        vision_out_test = self.vision_encoder.forward_features(dummy_input_vision)
        self.vision_feature_dim = vision_out_test.shape[-1] # e.g., 768 for ViT-Base
        self.num_vision_patches = vision_out_test.shape[1] # e.g., 196+1 for ViT-B/16 on 224px

        # Freeze Vision Encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Adapter: Projects vision features to LLM's hidden dimension
        # The number of output tokens from adapter will be self.num_vision_patches
        self.adapter = MLPAdapter(self.vision_feature_dim, self.llm_hidden_size)

        # Text Embeddings (for our BPE vocabulary)
        self.text_embedding = nn.Embedding(self.bpe_vocab_size, self.llm_embed_dim)
        
        # LLM (GPT-2 body)
        # We use GPT2Config to ensure hidden_size matches, but load pretrained weights for blocks.
        config = GPT2Config.from_pretrained(gpt2_model_name, n_embd=self.llm_hidden_size, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        # We only need the transformer blocks (h attribute of GPT2Model)
        # The GPT2Model includes wte, wpe, drop, h, ln_f.
        # We will use our own text_embedding, and a custom final LM head.
        # So, we take gpt2.h (transformer blocks) and gpt2.ln_f (final layer norm)
        gpt2 = GPT2Model.from_pretrained(gpt2_model_name, config=config)
        self.gpt2_blocks = gpt2.h
        self.gpt2_ln_f = gpt2.ln_f
        # self.gpt2_wpe = gpt2.wpe # Positional embeddings, if needed explicitly. GPT2 blocks handle it.

        # Freeze LLM parts
        for param in self.gpt2_blocks.parameters():
            param.requires_grad = False
        for param in self.gpt2_ln_f.parameters():
            param.requires_grad = False
        # if self.gpt2_wpe:
        #     for param in self.gpt2_wpe.parameters():
        #         param.requires_grad = False

        # Output LM Head (maps LLM hidden states to our BPE vocab)
        self.lm_head = nn.Linear(self.llm_hidden_size, self.bpe_vocab_size)

        self.image_size = dummy_img_size


    def forward(self, images, captions_input_ids, attention_mask=None):
        """
        images: (B, C, H, W)
        captions_input_ids: (B, S_text) - token IDs from our BPE tokenizer (input/shifted right)
        attention_mask: (B, S_combined) - mask for combined vision+text sequence
        """
        batch_size = images.shape[0]
        device = images.device

        # Image processing
        if images.shape[2:] != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode='bilinear', align_corners=False)
        
        vision_features_raw = self.vision_encoder.forward_features(images) # (B, NumPatches, D_vision)
        vision_embeddings = self.adapter(vision_features_raw) # (B, NumPatches, D_llm_hidden)
        
        # Text processing
        text_embeddings = self.text_embedding(captions_input_ids) # (B, S_text, D_llm_embed)
        
        # Concatenate vision and text embeddings
        # input_embeddings: (B, NumPatches + S_text, D_llm_hidden)
        input_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)

        # If caption_input_ids were padded, create corresponding attention mask for GPT-2
        # GPT-2's forward pass can take `attention_mask`
        # Vision tokens are always attended to. Text tokens are attended to based on padding.
        if attention_mask is None:
            vision_att_mask = torch.ones(batch_size, self.num_vision_patches, dtype=torch.long, device=device)
            # Create text attention mask based on padding (assuming pad_token_id=0 for BPE)
            # Here, captions_input_ids are inputs for teacher forcing, so they might have padding.
            pad_token_id = self.tokenizer.token_to_id(PAD_TOKEN) if self.tokenizer else 0
            text_att_mask = (captions_input_ids != pad_token_id).long().to(device)
            attention_mask = torch.cat([vision_att_mask, text_att_mask], dim=1) # (B, NumPatches + S_text)

        # GPT-2 processing (only transformer blocks)
        # GPT2Model's forward pass handles positional embeddings and layer norm.
        # Since we are using gpt2.h directly, we might need to handle positional embeddings
        # and initial dropout if they are not part of the blocks themselves.
        # GPT2Model does: wte + wpe -> drop -> h -> ln_f
        # We do: our_text_emb, concat with vision_emb.
        # Positional embeddings: GPT2 blocks often have their own relative/absolute pos enc.
        # For simplicity, let's assume the blocks handle positions or we add them.
        # GPT2 HF implementation adds positional embeddings *before* the blocks.
        # Let's add explicit positional encoding for the combined sequence.
        
        # This is a simplified view. Proper positional embeddings are crucial.
        # GPT2 uses learned positional embeddings (self.gpt2_wpe)
        # seq_len = input_embeddings.size(1)
        # position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0) # (1, S)
        # positional_embeddings = self.gpt2_wpe(position_ids) # (1, S, D_llm_hidden)
        # hidden_states = input_embeddings + positional_embeddings
        # hidden_states = self.gpt2.drop(hidden_states) # if gpt2.drop exists and is separate

        # The `GPT2Model.forward` has `past_key_values`, `attention_mask`, `token_type_ids`, `position_ids`, etc.
        # The `gpt2.h` (Block array) expects `hidden_states, layer_past, attention_mask, head_mask, ...`
        # It's simpler to pass `input_embeds` to a full `GPT2Model` instance and then slice its output,
        # or reconstruct its input stage carefully.

        # Simplified: pass embeddings directly to blocks, assuming they handle positions or are position-agnostic enough for this demo
        # The attention_mask needs to be in the format expected by GPT-2's MultiHeadAttention (usually for causal masking too)
        # For cross-attention like behavior (vision influencing text), the mask should allow it.
        # Standard causal mask for text, full attention for vision parts.
        
        # `attention_mask` for HF GPT2 blocks: (batch, 1, seq_len, seq_len) or (batch, seq_len) for padding
        # Let's use the padding mask, and causal masking is handled internally for decoder-style GPT2.
        # The provided attention_mask (B, S_combined) should be extended for causal masking if needed.
        # For GPT2, causal masking is default. This attention_mask handles padding.
        
        # GPT2 blocks expect `hidden_states` as input.
        transformer_outputs = self.gpt2_blocks(
            inputs_embeds=input_embeddings, # This is not a direct arg for `nn.ModuleList` of blocks
                                            # We need to iterate or use the full GPT2Model logic
            attention_mask=attention_mask   # Also needs to be passed correctly to each block's attention
        )
        # Correct way to use GPT2 blocks:
        hidden_states = input_embeddings # Potentially add positional embeddings here
        # Example of adding positional embeddings if `self.gpt2_wpe` was kept:
        # seq_len = input_embeddings.size(1)
        # position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).view(-1, seq_len)
        # hidden_states = hidden_states + self.gpt2_wpe(position_ids)
        
        # Iterate through GPT-2 blocks
        for block in self.gpt2_blocks:
            # Each block might return a tuple (hidden_state, present_key_value)
            # The `attention_mask` needs to be formatted for the attention mechanism.
            # For HF's GPT2Attention, a 2D mask (batch_size, sequence_length) is expanded to
            # (batch_size, 1, tgt_sequence_length, src_sequence_length)
            # Causal masking is applied internally if `is_decoder=True` or `causal=True`.
            # The mask we built handles padding.
            block_outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states = block_outputs[0]

        hidden_states = self.gpt2_ln_f(hidden_states) # Final LayerNorm

        # We only want to predict for the text part.
        # So, slice the output corresponding to text tokens.
        # hidden_states is (B, NumPatches + S_text, D_llm_hidden)
        text_hidden_states = hidden_states[:, self.num_vision_patches:, :] # (B, S_text, D_llm_hidden)
        
        logits = self.lm_head(text_hidden_states) # (B, S_text, bpe_vocab_size)
        return logits

    def generate_caption(self, image_tensor, max_len=50, device='cpu'):
        self.eval()
        if self.tokenizer is None:
            raise ValueError("BPE Tokenizer not set for MLLM generation.")

        sos_idx = self.tokenizer.token_to_id(SOS_TOKEN)
        eos_idx = self.tokenizer.token_to_id(EOS_TOKEN)
        pad_idx = self.tokenizer.token_to_id(PAD_TOKEN)

        batch_size = image_tensor.shape[0] # Should be 1 for generation
        if batch_size != 1: raise ValueError("Generation supports batch_size=1 only.")

        with torch.no_grad():
            # Image processing
            if image_tensor.shape[2:] != self.image_size:
                image_tensor = F.interpolate(image_tensor.to(device), size=self.image_size, mode='bilinear', align_corners=False)
            else:
                image_tensor = image_tensor.to(device)

            vision_features_raw = self.vision_encoder.forward_features(image_tensor)
            vision_embeddings = self.adapter(vision_features_raw) # (1, NumPatches, D_llm_hidden)

            # Start with SOS token for text part
            current_text_ids = torch.LongTensor([[sos_idx]]).to(device) # (1, 1)
            generated_ids_list = [sos_idx]

            for _ in range(max_len -1):
                text_embeddings = self.text_embedding(current_text_ids) # (1, current_S_text, D_llm_embed)
                
                input_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1) # (1, NumPatches + current_S_text, D_llm_hidden)
                
                # Attention mask for combined sequence
                # Vision part always attended, text part causally and based on current length
                # num_total_tokens = input_embeddings.shape[1]
                # att_mask = torch.ones(1, num_total_tokens, device=device) # Simplified for generation (no padding issues)
                
                # More careful attention mask handling might be needed if GPT-2 blocks expect specific format
                # For generation, causal masking is key for the text part.
                # The GPT-2 blocks internally handle causal masking for autoregressive generation.
                # The `attention_mask` here is primarily for padding, which isn't an issue in step-by-step generation.
                
                hidden_states = input_embeddings
                # Position embeddings if needed explicitly
                # seq_len = input_embeddings.size(1)
                # position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
                # hidden_states = hidden_states + self.gpt2_wpe(position_ids)

                for block in self.gpt2_blocks:
                    # For generation, `use_cache=True` and `past_key_values` would optimize this.
                    # Simplified: no cache used.
                    block_outputs = block(hidden_states) # No explicit mask needed if causal is internal and no padding
                    hidden_states = block_outputs[0]
                
                hidden_states = self.gpt2_ln_f(hidden_states)

                # Get the hidden state for the *last text token*
                last_text_hidden_state = hidden_states[:, -1, :] # (1, D_llm_hidden)
                
                logits = self.lm_head(last_text_hidden_state) # (1, bpe_vocab_size)
                predicted_idx = logits.argmax(dim=-1) # (1)
                
                generated_ids_list.append(predicted_idx.item())
                if predicted_idx.item() == eos_idx:
                    break
                
                # Append new token to current_text_ids for next iteration
                current_text_ids = torch.cat([current_text_ids, predicted_idx.unsqueeze(1)], dim=1)
        
        return self.tokenizer.decode(generated_ids_list, skip_special_tokens=True)


# Self-test
if __name__ == "__main__":
    print("Testing model_mllm.py...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy params
    bpe_vocab_s = 1000 # Our custom vocab size
    llm_embed_d = 768 # Should match GPT-2 hidden size for simplicity
    llm_hidden_s = 768 # GPT-2 'gpt2' (124M) uses 768
    gpt2_name = "gpt2" # Smallest GPT-2
    vit_name = "vit_tiny_patch16_224" # Small ViT
    max_s_len = 30 # Max combined vision + text tokens in fwd pass
    batch_s = 2
    
    # Mock BPE tokenizer for generation test
    class MockBpeTokenizer:
        def token_to_id(self, token):
            if token == SOS_TOKEN: return 1
            if token == EOS_TOKEN: return 2
            if token == PAD_TOKEN: return 0
            return 3 # unk
        def decode(self, ids, skip_special_tokens=True):
            map_dict = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: "word"}
            tokens = []
            for _id in ids:
                if skip_special_tokens and _id in [0,1,2]:
                    if _id == 1 and not tokens: pass
                    else: continue
                tokens.append(map_dict.get(_id, "unknown"))
            return " ".join(tokens)
        def get_vocab_size(self): return bpe_vocab_s

    mock_bpe_tokenizer = MockBpeTokenizer()

    try:
        model = MLLM(
            bpe_vocab_size=bpe_vocab_s,
            llm_embed_dim=llm_embed_d, # Will be set to llm_hidden_s if different
            llm_hidden_size=llm_hidden_s,
            gpt2_model_name=gpt2_name,
            vision_encoder_name=vit_name,
            max_seq_len=max_s_len, # Used for PE sizing if explicit
            tokenizer=mock_bpe_tokenizer
        ).to(device)

        # Unfreeze adapter and lm_head for test (normally done in trainer)
        for param in model.adapter.parameters():
            param.requires_grad = True
        for param in model.text_embedding.parameters(): # BPE embeddings also need training
             param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
        
        # Original image size 48x48, model will resize to ViT's expected input (e.g., 224x224)
        dummy_images = torch.randn(batch_s, 3, 48, 48).to(device) 
        
        # Max text length for this batch, e.g. 15 BPE tokens
        # captions_input_ids are for teacher forcing, so they are like (SOS, t1, t2, ..., tn)
        # They will be padded to a fixed length in a batch (e.g. max_s_len - num_vision_patches)
        # For testing, let S_text = 15
        s_text = 15 
        dummy_captions_in = torch.randint(0, bpe_vocab_s, (batch_s, s_text)).to(device)

        # Check total sequence length
        num_vision_tokens_actual = model.num_vision_patches
        if num_vision_tokens_actual + s_text > max_s_len:
            print(f"Warning: num_vision_tokens ({num_vision_tokens_actual}) + s_text ({s_text}) > max_seq_len ({max_s_len}). Truncate text.")
            s_text = max_s_len - num_vision_tokens_actual
            dummy_captions_in = dummy_captions_in[:, :s_text]


        output_logits = model(dummy_images, dummy_captions_in)
        print(f"MLLM output shape: {output_logits.shape}") # (batch_s, s_text, bpe_vocab_s)
        assert output_logits.shape == (batch_s, s_text, bpe_vocab_s)
        
        # Test generation
        single_image = torch.randn(1, 3, 48, 48).to(device)
        generated_caption = model.generate_caption(single_image, max_len=15, device=device) # Max_len for generated text part
        print(f"Generated MLLM caption (dummy): '{generated_caption}'")
        assert isinstance(generated_caption, str)

        # Check which parameters are trainable
        print("\nTrainable parameters for MLLM:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        
        # Expected trainable: adapter, text_embedding, lm_head
        assert model.adapter.projection.weight.requires_grad
        assert model.text_embedding.weight.requires_grad
        assert model.lm_head.weight.requires_grad
        # Expected frozen: vision_encoder, gpt2_blocks, gpt2_ln_f
        assert not model.vision_encoder.patch_embed.proj.weight.requires_grad
        assert not model.gpt2_blocks[0].attn.c_attn.weight.requires_grad
        assert not model.gpt2_ln_f.weight.requires_grad


        print("model_mllm.py tests passed!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during model_mllm.py test: {e}")
        print("This might be due to model download/compatibility (timm, transformers) or architectural assumptions.")
        print("Ensure 'timm', 'transformers', 'einops' are installed and an internet connection is available for model download.")
