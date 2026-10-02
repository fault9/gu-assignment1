"""
For each model, evaluation.py should load the trained model, and for each language compute BPC.
"""

from pathlib import Path

import torch
import numpy as np
from torch.nn.functional import cross_entropy
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
    ids = torch.tensor(np.concatenate([[EOS], stream], dtype=torch.long, device=DEVICE)

    losses = [] # save each loss here

    with torch.no_grad(): # testing, not learning
        # read stream in context pieces
        for start in range (0, len(stream), model.context):
            end = min(start + model.context, len(stream))
            inputs = ids[start:end] # what model reads
            targets = ids[start+1:end+1] # what model should guess - one step ahead
    
