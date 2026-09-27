import csv
from pathlib import Path

import sentencepiece as spm

from src.data import load_split, n_chars, LANGS

TOKENIZERS = ("char", "bpe_2k", "bpe_10k")
ARTIFACT_DIR = Path("artifacts/tokenizers")
OUT_PATH = Path("results/tokenizer_stats.csv")

def load_tokenizer(name):
    """Load one trained SentencePiece model by name."""
    return spm.SentencePieceProcessor(model_file=str(ARTIFACT_DIR / f"{name}.model"))

def tokenizer_stats(split="valid"):
    """Per (tokenizer, language): tokens needed, tokens/sentence, chars/token."""
    rows = []
    for name in TOKENIZERS:
        sp = load_tokenizer(name)
        for lang in LANGS:
            lines = load_split(split, lang)
            chars = n_chars(lines)
            tokens = sum(len(sp.encode(line)) for line in lines)
            rows.append({
                "tokenizer": name,
                "lang": lang,
                "vocab_size": sp.get_piece_size(),
                "sentences": len(lines),
                "chars": chars,
                "tokens": tokens,
                "tokens_per_sentence": round(tokens / len(lines), 1),
                "chars_per_token": round(chars / tokens, 3),
            })
    return rows

def write_stats(rows, path=OUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")

def print_table(rows):
    """Readable summary: chars per token, tokenizer x language."""
    print(f"\n{'chars/token':<14}" + "".join(f"{l:>10}" for l in LANGS))
    for name in TOKENIZERS:
        vals = [r["chars_per_token"] for r in rows if r["tokenizer"] == name]
        print(f"{name:<14}" + "".join(f"{v:>10.3f}" for v in vals))

    print(f"\n{'total tokens':<14}" + "".join(f"{l:>10}" for l in LANGS))
    for name in TOKENIZERS:
        vals = [r["tokens"] for r in rows if r["tokenizer"] == name]
        print(f"{name:<14}" + "".join(f"{v:>10,}" for v in vals))

if __name__ == "__main__":
    rows = tokenizer_stats("valid")
    write_stats(rows)
    print_table(rows)
