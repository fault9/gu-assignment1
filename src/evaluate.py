"""
For each model, evaluation.py should load the trained model, and for each language compute BPC.
"""

from pathlib import Path

import torch
import numpy as np
from torch.nn.functional import cross_entropy
import math

from src.data import load_split, n_chars
from src.dataset import EOS
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

