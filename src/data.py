from pathlib import Path
import os
import csv

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/raw"))
LANGS = ("en", "tr", "zh")
SPLITS = ("train", "valid", "test")

def load_split(split: str, lang: str) -> list[str]:
    """Load one split of one language as a list of sentences."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}, expected one of {LANGS}")

    path = DATA_ROOT / split / f"{lang}.txt"
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def n_chars(lines: list[str]) -> int:
    """Total Unicode code points across the sentences (newlines excluded)."""
    return sum(len(line) for line in lines)

def corpus_stats() -> list[dict]:
    """Per (split, language): sentences, characters, distinct characters."""
    rows = []
    for split in SPLITS:
        for lang in LANGS: 
            lines = load_split(split, lang)
            chars = n_chars(lines)
            rows.append({
                "split": split, 
                "lang": lang, 
                "sentences": len(lines),
                "chars": chars,
                "distinct_chars": len({ch for line in lines for ch in line}),
                "chars_per_sentence": round(chars/len(lines))
            })
    return rows

def write_stats(path: Path = Path("results/corpus_stats.csv")) -> None:
    rows = corpus_stats()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")