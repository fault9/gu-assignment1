"""WHICH TOKENS BELONG TO WHICH LANGUAGE?"""

import numpy as np
from src.data import LANGS
from src.dataset import token_stream
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

    return counts 