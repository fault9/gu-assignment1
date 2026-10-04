"""Error analysis - test sentences where the models disagree most."""

import csv

from src.data import load_split
from src.analysis.token_stats import load_tokenizer

TOKENIZERS = ("char", "bpe_2k", "bpe_10k")

# picked from results/sentence_scores.csv: sentences with the biggest
# difference between the char and bpe_10k models
PICKS = [("zh", 471), ("zh", 5152), ("tr", 2925), ("en", 3445)]


def load_scores():
    """BPC of every test sentence, looked up by (lang, line)."""
    scores = {}
    with open("results/sentence_scores.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[(row["lang"], int(row["line"]))] = row
    return scores


if __name__ == "__main__":
    scores = load_scores()

    for lang, line in PICKS:
        sentence = load_split("test", lang)[line]
        row = scores[(lang, line)]
        print(f"{lang} {line}: {sentence}")

        for name in TOKENIZERS:
            pieces = load_tokenizer(name).encode(sentence, out_type=str)
            print(f"  {name:<8} bpc {row[name]}  {len(pieces)} tokens  {'|'.join(pieces)}")
        print()