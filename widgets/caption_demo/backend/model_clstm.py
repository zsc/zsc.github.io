import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.data_utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN # For token IDs in generation

class ConvEncoder(nn.Module):
    def __init__(self, encoded_size=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) # 48x48 -> 24x24
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 24x24 -> 12x12
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 12x12 -> 6x6
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 6x6 -> 3x3
        self.bn4 = nn.BatchNorm2d(256)
        # Global average pooling or flatten and linear
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, encoded_size)

    def forward(self, images): # images: (batch, 3, H, W)
        x = F.relu(self.bn1(self.conv1(images)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x) # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 256)
        features = self.fc(x) # (batch, encoded_size)
        return features

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions, lengths=None):
        """
        features: (batch_size, feature_dim) - from encoder (used as initial hidden state)
        captions: (batch_size, seq_len) - ground truth captions
        lengths: (batch_size) - actual lengths of captions (optional, for packed sequence)
        """
        embeddings = self.dropout(self.embedding(captions)) # (batch_size, seq_len, embed_size)
        
        # Initialize LSTM state with image features
        # We need to map `features` (encoder_output_dim) to `hidden_size * num_layers` for h0 and c0
        # For simplicity, if feature_dim matches hidden_size, we can use it directly.
        # Otherwise, add a linear layer. Here, assume feature_dim is adaptable.
        # Let's assume features are used to initialize only the first hidden state layer.
        # A common way is to make feature_dim == hidden_size.
        # Or, project features to hidden_size for h0 and c0
        # For this example, let's assume features are the initial hidden state for LSTM
        # If using features directly, ensure its size matches hidden_size
        # features need to be shaped to (num_layers, batch_size, hidden_size)
        
        # A simple projection:
        # if features.size(1) != self.hidden_size:
        #    raise ValueError("Feature size from encoder must match decoder hidden size for this simple init.")
        # h0 = features.unsqueeze(0) # (1, batch_size, hidden_size) - for num_layers=1
        # c0 = torch.zeros_like(h0)  # (1, batch_size, hidden_size)

        # Better: LSTM expects (input, (h_0, c_0))
        # Image features are used as the *first input* to the LSTM or to initialize hidden states.
        # Let's treat image features as part of the sequence or use them to condition h0, c0.
        # A common approach: concatenate features with each word embedding, or use features to initialize (h0,c0).
        # To initialize h0,c0:
        #   Need linear layers if feature_dim != hidden_size or if num_layers > 1
        # For this "from scratch" version, let's use the image features as the initial input to the LSTM.
        # We can prepend it to the embeddings sequence, or make the LSTMCell accept it.
        # Simpler: image features project to h0 and c0. Assume features dimension is hidden_size.
        
        # If features are (batch, hidden_size) and num_layers=1:
        h0 = features.unsqueeze(0) # (1, batch, hidden_size)
        c0 = torch.zeros_like(h0) # (1, batch, hidden_size)

        # If features are (batch, embed_size) and should be the first token:
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) # (batch, seq_len+1, embed_size)
        # This makes captions argument for teacher forcing shifted by 1 if so.
        # Let's stick to (h0,c0) initialization for now.

        if lengths is not None: # TODO: This part requires more careful handling with PackedSequence
            # embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
            pass # Skipping packed_sequence for simplicity in this example

        lstm_out, _ = self.lstm(embeddings, (h0, c0)) # lstm_out: (batch, seq_len, hidden_size)
        
        # if lengths is not None:
            # lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        outputs = self.fc_out(lstm_out) # (batch_size, seq_len, vocab_size)
        return outputs

class ConvLSTMModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, 
                 encoder_feature_size=512, num_layers=1, dropout=0.1, tokenizer=None):
        super().__init__()
        self.encoder = ConvEncoder(encoded_size=encoder_feature_size)
        # The LSTM decoder's hidden_size should match encoder_feature_size if we directly use features for h0/c0
        # Or, we add a projection layer. For this example, let's assume they match.
        if encoder_feature_size != hidden_size:
            print(f"Warning: ConvLSTMModel encoder_feature_size ({encoder_feature_size}) != decoder hidden_size ({hidden_size}). Ensure proper adaptation or matching sizes.")
            # Add a projection layer if needed, or adjust hidden_size to match encoder_feature_size
            # For now, we'll assume they are meant to be the same, and hidden_size is the primary driver.
            # So, the encoder output should be hidden_size
            self.encoder = ConvEncoder(encoded_size=hidden_size) # Make encoder output match LSTM hidden
            
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers, dropout)
        self.tokenizer = tokenizer # Store for generation

    def forward(self, images, captions_input):
        features = self.encoder(images)
        outputs = self.decoder(features, captions_input)
        return outputs

    def generate_caption(self, image_tensor, max_len=50, device='cpu'):
        """
        Generate caption for a single image.
        image_tensor: (1, C, H, W)
        """
        self.eval()
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set for generation.")

        sos_idx = self.tokenizer.token_to_id(SOS_TOKEN)
        eos_idx = self.tokenizer.token_to_id(EOS_TOKEN)
        pad_idx = self.tokenizer.token_to_id(PAD_TOKEN)

        generated_ids = [sos_idx]
        
        with torch.no_grad():
            features = self.encoder(image_tensor.to(device)) # (1, hidden_size)
            
            # Prepare initial hidden and cell states for LSTM
            # Assuming encoder output size matches decoder hidden size and num_layers=1 for decoder
            h_state = features.unsqueeze(0) # (1, 1, hidden_size)
            c_state = torch.zeros_like(h_state).to(device) # (1, 1, hidden_size)

            current_word_idx = torch.LongTensor([sos_idx]).to(device) # (1)

            for _ in range(max_len):
                # embed: (1, 1, embed_size)
                embedded_word = self.decoder.embedding(current_word_idx).unsqueeze(1) 
                
                # lstm_out: (1, 1, hidden_size), (h_state, c_state)
                lstm_out, (h_state, c_state) = self.decoder.lstm(embedded_word, (h_state, c_state))
                
                # output: (1, vocab_size)
                output_logits = self.decoder.fc_out(lstm_out.squeeze(1))
                
                predicted_idx = output_logits.argmax(1) # Greedy search
                
                if predicted_idx.item() == eos_idx:
                    break
                generated_ids.append(predicted_idx.item())
                current_word_idx = predicted_idx
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# Self-test
if __name__ == "__main__":
    print("Testing model_clstm.py...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy params
    embed_s = 256
    hidden_s = 512 # This should match encoder_feature_size for simple h0/c0 init
    vocab_s = 1000 
    img_h, img_w = 48, 48
    batch_s = 4
    seq_l = 20

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
            for _id in ids:
                if skip_special_tokens and _id in [0,1,2]:
                    if _id == 1 and not tokens: # Keep first SOS if not skipping
                        pass
                    else:
                        continue
                tokens.append(map_dict.get(_id, "unknown"))
            return " ".join(tokens)


    # Test Encoder
    encoder = ConvEncoder(encoded_size=hidden_s).to(device)
    dummy_images = torch.randn(batch_s, 3, img_h, img_w).to(device)
    features = encoder(dummy_images)
    print(f"Encoder output shape: {features.shape}") # (batch_s, hidden_s)
    assert features.shape == (batch_s, hidden_s)

    # Test Decoder
    decoder = LSTMDecoder(embed_s, hidden_s, vocab_s).to(device)
    dummy_captions = torch.randint(0, vocab_s, (batch_s, seq_l)).to(device)
    # features from encoder are (batch_s, hidden_s), decoder expects this for h0/c0 features
    output_logits = decoder(features, dummy_captions)
    print(f"Decoder output shape: {output_logits.shape}") # (batch_s, seq_l, vocab_s)
    assert output_logits.shape == (batch_s, seq_l, vocab_s)

    # Test Full Model
    model = ConvLSTMModel(embed_s, hidden_s, vocab_s, encoder_feature_size=hidden_s, tokenizer=MockTokenizer()).to(device)
    output_logits_full = model(dummy_images, dummy_captions)
    print(f"Full model output shape: {output_logits_full.shape}")
    assert output_logits_full.shape == (batch_s, seq_l, vocab_s)
    
    # Test generation
    single_image = torch.randn(1, 3, img_h, img_w).to(device)
    generated_caption = model.generate_caption(single_image, max_len=10, device=device)
    print(f"Generated caption (dummy): '{generated_caption}'")
    assert isinstance(generated_caption, str)

    print("model_clstm.py tests passed!")
