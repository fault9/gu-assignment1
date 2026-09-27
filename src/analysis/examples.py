from pathlib import Path

from src.data import load_split
from src.analysis.token_stats import load_tokenizer, TOKENIZERS

OUT_PATH = Path("results/example_tokenizations.md")

EXAMPLES = [
    ("en", 1),    # rare proper nouns: DMK, Tamil Nadu
    ("tr", 13),   # heavy suffixation: kalıbının, kullanıldığına, rastlanabilir
    ("zh", 3),    # mixed script: Chinese with Latin (IJF) and digits (12)
]

def write_examples():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        print("# Example tokenizations\n", file=f)

        for lang, idx in EXAMPLES:
            sentence = load_split("valid", lang)[idx]
            print(f"{lang} (valid line {idx}) — {len(sentence)} characters\n", file=f)
            print(f"{sentence}\n", file=f)
            for name in TOKENIZERS:
                sp = load_tokenizer(name)
                pieces = sp.encode(sentence, out_type=str)
                print(f"**{name}** — {len(pieces)} tokens, "
                      f"{len(sentence)/len(pieces):.2f} chars/token\n", file=f)
                print("```", file=f)
                print("|".join(pieces), file=f)
                print("```\n", file=f)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    write_examples()