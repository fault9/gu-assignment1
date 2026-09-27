from pathlib import Path
import sentencepiece as spm

from src.data import DATA_ROOT

ARTIFACT_DIR = Path("artifacts/tokenizers")


def train_tokenizer(vocab_size: int, character_coverage: float, name: str, model_type = "bpe") -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = ARTIFACT_DIR/name

    spm.SentencePieceTrainer.train(
        input=str(DATA_ROOT / "tokenizer" / "balanced.txt"),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=True,
        normalization_rule_name="identity",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )
    
    model = prefix.with_suffix(".model")
    sp = spm.SentencePieceProcessor(model_file=str(model))
    print(f"{name}: requested {vocab_size}, realised {sp.get_piece_size()}")
    return model

if __name__ == "__main__":
    train_tokenizer(vocab_size=2500,  character_coverage=0.98, name="bpe_2k")
    train_tokenizer(vocab_size=10000, character_coverage=0.98, name="bpe_10k")
    train_tokenizer(vocab_size=9702,  character_coverage=1.0,  name="char", model_type="char")
    