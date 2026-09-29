import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import LANGS  

RESULTS_DIR = Path("results/runs")
FIGURES_DIR = Path("figures")
TOKENIZERS = ("char", "bpe_2k", "bpe_10k")

def load_results(tokenizer_name):
    """Read one run log and return list of dicts"""
    path = RESULTS_DIR/tokenizer_name/"log.jsonl"
    with path.open() as f: 
        return [json.loads(line) for line in f]
    
def plot_learning_curves():
    """Function for comparing conditions, and showing each tokenizer separately"""

    FIGURES_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    #left panel
    for name in TOKENIZERS:
        rows = load_results(name)
        axes[0].plot([r["step"] for r in rows],
                     [r["val_mean"] for r in rows], label=name)
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("optimiser updates")
    axes[0].set_ylabel("cross-entropy (nats/token)")
    axes[0].legend()

    # one panel per tokenizer, split by language
    for ax, name in zip(axes[1:], TOKENIZERS):
        rows = load_results(name)
        steps = [r["step"] for r in rows]
        for lang in LANGS:
            ax.plot(steps, [r[f"val_{lang}"] for r in rows], label=lang)
        ax.plot(steps, [r["train_loss"] for r in rows],
                linestyle=":", color="grey", label="train")
        ax.set_title(name)
        ax.set_xlabel("optimiser updates")
        ax.legend()

    fig.tight_layout()
    out = FIGURES_DIR / "learning_curves.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_learning_curves()