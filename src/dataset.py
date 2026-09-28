from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from src.data import load_split, LANGS

TOKENIZED_DIR = Path("data/tokenized")
ARTIFACT_DIR = Path("artifacts/tokenizers")

EOS = 3

def token_stream(token_name, split, lang):
    """Tokenize one split of language into an id array.
    
    Returns one long array, e.g [412, 88, 7, 3, 901, ...] 
    with every sentence followed by EOS
    """
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    path = TOKENIZED_DIR / f"{token_name}_{split}_{lang}.npy"
    # If already tokenized, load it
    if path.exists():
        return np.load(path)
    
    sp = spm.SentencePieceProcessor(model_file=str(ARTIFACT_DIR/f"{token_name}.model"))

    ids = []
    for line in load_split(split, lang):
        ids.extend(sp.encode(line)) # add all the sentence's ids
        ids.append(EOS) # add one id, for marking boundary

    # save
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.array(ids, dtype=np.int32)
    np.save(path, arr)
    return arr

def load_streams(token_name, split):
    """A list of three arrays, one per language: [en_ids, tr_ids, zh_ids]"""
    return [token_stream(token_name, split, lang) for lang in LANGS]

def get_batch(streams, rng, batch_size=64, context=256, device="cpu"):
    """Take batch_size random windows of text. Returns tuple (inputs, targets).

    The model is predicting the next token, so target is simply the input shifted one position ot the right
    """

    inputs = [] 
    targets = [] 

    for _ in range(batch_size):
        #pick one of the langs at random, whole windo wfrom one language
        stream = streams[rng.integers(len(streams))]

        #pick random place to start reading -context-1 so there is room for full window plus position shift
        i = rng.integers(len(stream)-context-1)

        inputs.append(stream[i : i + context])
        targets.append(stream[i+1:i+context+1]) #shifted by 1

    #convert lists of np slices to tensors for lookup. hence scalar type Long
    x = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    y = torch.tensor(np.array(targets), dtype=torch.long, device=device)
    return x, y

if __name__ == "__main__":
    # how many tokens does each tokenizer turn the training data into?

    for token_name in ("char", "bpe_2k", "bpe_10k"):
        streams = load_streams(token_name, "train")
        print(f"{token_name:<9} {sum(len(s) for s in streams):>11,} training tokens")
    