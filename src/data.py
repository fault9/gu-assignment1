from pathlib import Path
import os

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