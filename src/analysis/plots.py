import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from src.data import LANGS  

RESULTS_DIR = Path("results/runs")
FIGURES_DIR = Path("figures")
TOKENIZERS = ("char", "bpe2k", "bpe_10k")

def load_results(tokenizer_name):
    """Read one run log and return list of dicts"""
    path = RESULTS_DIR/tokenizer_name/"log.jsonl"
    with path.open() as f: 
        return [json.lods(line) for line in f]
    
def plot_learning_curves():
    """Function for comparing conditions, and showing each tokenizer separately"""

    FIGURES_DIR.mkdir(exist_ok=True)
    fig, panels = plt.subplots(1, 4, figsize=(18, 4))

    #left panel
    for name in TOKENIZERS:
        rows = load_results(name)
        panels[0].plot([r["step"] for r in rows],
                     [r["val_mean"] for r in rows], label=name)
    panels[0].set_title("Validation loss")
    panels[0].set_xlabel("optimiser updates")