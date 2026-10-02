"""
For each model, evaluation.py should load the trained model, and for each language compute BPC.
"""

from pathlib import Path

import torch

from src.model import TransformerLM
from src.train import DEVICE, RESULTS_DIR

TOKENIZERS = ("char", "bpe_2k", "bpe_10k")

def load_model(tokenizer):
    """
    Rebuilds the model and uses same weights as used during training
    """

    checkpoint = torch.load(RESULTS_DIR/tokenizer/"best.pt", weights_only=True)

    model = TransformerLM(checkpoint["vocab_size"], context=checkpoint["context"])
    model.load_state_dict(checkpoint["model"]) # copies trained weights to the new model
    model.to(DEVICE) # moves the model to the GPU
    model.eval() # begins eval mode, no dropout

    return model