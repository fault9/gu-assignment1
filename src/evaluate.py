"""
For each model, evaluation.py should load the trained model, and for each language compute BPC.
"""

from pathlib import Path

import torch
import numpy as np
from torch.nn.functional import cross_entropy
import math
import csv


from src.data import load_split, n_chars, LANGS
from src.dataset import EOS, token_stream
from src.model import TransformerLM
from src.train import DEVICE, RESULTS_DIR

TOKENIZERS = ("char", "bpe_2k", "bpe_10k")

def load_model(tokenizer):
    """
    Rebuilds the model and uses same weights as used during training
    """

    checkpoint = torch.load(RESULTS_DIR/tokenizer/"best.pt", weights_only=True)

    model = TransformerLM(checkpoint["vocab_size"], context=checkpoint["context"]) # model context = 256 
    model.load_state_dict(checkpoint["model"]) # copies trained weights to the new model
    model.to(DEVICE) # moves the model to the GPU
    model.eval() # begins eval mode, no dropout

    return model

def token_losses (model, stream):
    """loss for every token in the stream"""
    ids = torch.tensor(np.concatenate([[EOS], stream]), dtype=torch.long, device=DEVICE)

    losses = [] # save each loss here

    with torch.no_grad(): # testing, not learning
        # read stream in context pieces
        for start in range (0, len(stream), model.context):
            end = min(start + model.context, len(stream))
            inputs = ids[start:end] # what model reads
            targets = ids[start+1:end+1] # what model should guess - one step ahead

            # model wants a batch of pieces, I have one piece: shape (256) -> (1, 256)
            predictions = model(inputs.unsqueeze(0))
            loss = cross_entropy(predictions[0], targets, reduction="none")
            losses.append(loss)

    # add all pieces together, move from gpu to cpu
    return torch.cat(losses).cpu().numpy()


def bits_per_char(losses, stream, lines):
    # losses are in nats, divide by log(2) to turn them into bits, 
    # does not count EOS tokens
    total_bits = losses[stream != EOS].sum() / math.log(2)
    chars = n_chars(lines)
    return total_bits / chars

def sentence_bpc(losses, stream, lines):
    """returns BPC per sentence, in same order as lines"""

    sentence_bits = [] # total bits of each sentence here
    
    bits = 0.0 # bits of sentence currently being read

    for loss, token in zip(losses, stream):
        if token == EOS:
            sentence_bits.append(bits)
            bits = 0.0 # reset
        else:
            bits += loss / math.log(2)

    # divide each sentence's bits by num of chars
    scores = []

    for bits, line in zip(sentence_bits, lines):
        chars = len(line)

        if chars == 0:
            chars = 1 #avoid zero division
        scores.append(bits/chars)

    return scores

UNK = 1

if __name__ == "__main__":
    results = []
    sentence_scores = {"en": {}, "tr": {}, "zh": {}}
    for tokenizer in TOKENIZERS:
        model = load_model(tokenizer)
        all_bits = 0  
        all_chars = 0 
        for lang in LANGS:
            lines = load_split("test", lang)
            stream = token_stream(tokenizer, "test", lang)
            losses = token_losses(model, stream)
            bpc = bits_per_char(losses, stream, lines)
            chars = n_chars(lines)
            results.append({
                "tokenizer": tokenizer,
                "lang": lang,
                "chars": chars,
                "tokens": int((stream != EOS).sum()),
                "unk": int((stream == UNK).sum()),
                "bpc": round(bpc, 3),
            })
            all_bits += bpc*chars
            all_chars += chars

            sentence_scores[lang][tokenizer] = sentence_bpc(losses, stream, lines)
            print(f"{tokenizer:<8} {lang}  bpc {bpc:.3f}")

            # all bits of all languages/all characters of all languages
        results.append({"tokenizer": tokenizer, "lang": "all",
                        "chars": all_chars, "tokens": "", "unk": "",
                        "bpc": round(all_bits / all_chars, 3)})
        print(f"{tokenizer:<8} all bpc {all_bits / all_chars:.3f}")

    with open("results/test_bpc.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("wrote results/test_bpc.csv")

    #one row per sentence
    with open("results/sentence_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lang", "line", "char", "bpe_2k", "bpe_10k", "sentence"])

        for lang in LANGS:
            lines = load_split("test", lang)
            for i in range(len(lines)):
                writer.writerow([lang, i,
                                round(sentence_scores[lang]["char"][i], 3),
                                round(sentence_scores[lang]["bpe_2k"][i], 3),
                                round(sentence_scores[lang]["bpe_10k"][i], 3),
                                lines[i]])
    print("wrote results/sentence_scores.csv")