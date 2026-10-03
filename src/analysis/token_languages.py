"""WHICH TOKENS BELONG TO WHICH LANGUAGE?"""

import numpy as np
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