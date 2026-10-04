"""WHICH TOKENS BELONG TO WHICH LANGUAGE?"""

import numpy as np
import csv
from collections import Counter
from pathlib import Path

from src.data import LANGS
from src.dataset import token_stream,EOS
from src.analysis.token_stats import load_tokenizer

BPE_TOKENIZERS = ("bpe_2k", "bpe_10k")

def count_tokens(tokenizer):
    """Retrieves how many times each token id is used in the training data per for each language"""

    # how many different tokens the tokenizer has 
    vocab_size = load_tokenizer(tokenizer).get_piece_size()

    # dict to map the langs
    counts = {} 

    for lang in LANGS:
        stream = token_stream(tokenizer, "train", lang)

        #bincount counts frequency of id appearance
        counts[lang] = np.bincount(stream, minlength=vocab_size)
        counts[lang][EOS] = 0

    return counts 

def language_shares(counts):
    """for each token, what part of it comes from en tr and zh"""

    #rate = count / all tokens of that lang
    rates = {}
    for lang in LANGS:
        rates[lang] = counts[lang] / counts[lang].sum()

    total = rates["en"] + rates["tr"] + rates["zh"]

    #tokens never used should be set to 1 so no zero division
    total[total==0] = 1

    #share = this language's rate / all rates
    shares = {}
    for lang in LANGS:
        shares[lang] = rates[lang] / total
    return shares

CUTOFF = 0.9 # cutoff for when a token belongs to a language (90% or more)
def label_tokens(counts, shares):
    vocab_size = len(counts["en"])
    labels = []

    for i in range(vocab_size):
        # how often this token is used in all three langs together
        used = counts["en"][i] + counts["tr"][i] + counts["zh"][i]

        if used == 0:
            labels.append("unused") # <unk> </s> etc
            continue

        label = "shared"
        for lang in LANGS:
            if shares[lang][i] >= CUTOFF:
                label = lang

        labels.append(label)

    return labels

def summarize(tokenizer):
    """How many tokens each language gets"""
    sp = load_tokenizer(tokenizer)
    counts = count_tokens(tokenizer)
    labels = label_tokens(counts, language_shares(counts))

    # how many tokens got each label
    label_counts = Counter(labels)
    row = {"tokenizer": tokenizer}
    for label in ("en", "tr", "zh", "shared", "unused"):
        row[label] = label_counts[label]

    # the 10 most used tokens of each label
    total = counts["en"] + counts["tr"] + counts["zh"]   
    most_used_first = total.argsort()[::-1]               
    for label in ("en", "tr", "zh", "shared"):
        examples = []
        for i in most_used_first:
            if labels[i] == label:
                examples.append(sp.id_to_piece(int(i)))
            if len(examples) == 10:
                break
        print(tokenizer, label, examples)

    return row


if __name__ == "__main__":
    rows = []
    for tokenizer in BPE_TOKENIZERS:
        rows.append(summarize(tokenizer))

    with open("results/token_languages.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(rows)
