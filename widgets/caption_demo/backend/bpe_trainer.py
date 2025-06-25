import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from backend.data_utils import load_captions, CAPTIONS_PATH, DATA_DIR, ensure_data_exists
from backend.data_utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

BPE_MODEL_DIR = "./backend/bpe_model"
TOKENIZER_FILE = os.path.join(BPE_MODEL_DIR, "tokenizer.json")

DEFAULT_VOCAB_SIZE = 10000

def train_bpe_tokenizer(captions_file: str, vocab_size: int, output_dir: str, tokenizer_filename: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, tokenizer_filename)

    captions = load_captions(captions_file)

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
        min_frequency=2 # Example, adjust as needed
    )
    
    tokenizer.train_from_iterator(captions, trainer=trainer)

    # Optional: Add post-processor for SOS/EOS if needed during generation,
    # but often handled manually in the model's generation loop or dataset.
    # tokenizer.post_processor = TemplateProcessing(
    #     single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
    #     special_tokens=[
    #         (SOS_TOKEN, tokenizer.token_to_id(SOS_TOKEN)),
    #         (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
    #     ],
    # )
    
    tokenizer.save(output_path)
    print(f"BPE tokenizer trained and saved to {output_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer

def load_tokenizer(tokenizer_path: str = TOKENIZER_FILE) -> Tokenizer:
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}. Please train it first.")
    return Tokenizer.from_file(tokenizer_path)

# Self-test
if __name__ == "__main__":
    print("Testing bpe_trainer.py...")
    ensure_data_exists() # Ensure captions file is available (even dummy)
    
    # Use a smaller vocab for quick test
    test_vocab_size = 500
    test_output_dir = os.path.join(BPE_MODEL_DIR, "test_bpe")
    test_tokenizer_file = "test_tokenizer.json"
    
    print(f"Training a test BPE tokenizer with vocab size {test_vocab_size}...")
    try:
        tokenizer = train_bpe_tokenizer(CAPTIONS_PATH, test_vocab_size, test_output_dir, test_tokenizer_file)
        
        # Test loading
        loaded_tokenizer = load_tokenizer(os.path.join(test_output_dir, test_tokenizer_file))
        assert loaded_tokenizer.get_vocab_size() <= test_vocab_size # Can be slightly smaller if not enough unique tokens
        
        # Test encoding
        sample_text = "This is a test sentence with some words."
        encoded = loaded_tokenizer.encode(sample_text)
        print(f"Encoded '{sample_text}': {encoded.ids} -> {encoded.tokens}")
        assert len(encoded.ids) > 0

        # Test special tokens
        assert loaded_tokenizer.token_to_id(PAD_TOKEN) is not None
        assert loaded_tokenizer.token_to_id(SOS_TOKEN) is not None
        assert loaded_tokenizer.token_to_id(EOS_TOKEN) is not None
        assert loaded_tokenizer.token_to_id(UNK_TOKEN) is not None

        print("bpe_trainer.py tests passed!")
    except FileNotFoundError as e:
        print(f"Skipping BPE trainer test due to missing file: {e}")
        print("This might be because the dummy caption file is empty or CAPTIONS_PATH is incorrect.")
    finally:
        # Clean up test files
        if os.path.exists(os.path.join(test_output_dir, test_tokenizer_file)):
            os.remove(os.path.join(test_output_dir, test_tokenizer_file))
        if os.path.exists(test_output_dir) and not os.listdir(test_output_dir): # if dir is empty
             os.rmdir(test_output_dir)
