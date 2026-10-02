import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from src.data import LANGS
from src.dataset import load_streams, get_batch
from src.model import TransformerLM

RESULTS_DIR = Path("results/runs")

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def evaluate(model, token_streams, sampler, num_batches=20,
             batch_size=64, context=256):
    """Measure loss on validation text the model has never trained on."""
    model.eval()                       # dropout off while measuring
    losses = {}

    with torch.no_grad():              # not learning, so skip gradient machinery
        for language, stream in zip(LANGS, token_streams):
            total_loss = 0.0

            for _ in range(num_batches):
                inputs, targets = get_batch([stream], sampler, batch_size,
                                            context, DEVICE)
                predictions = model(inputs)
                vocab_size = predictions.shape[-1]

                loss = cross_entropy(predictions.reshape(-1, vocab_size),
                                     targets.reshape(-1))
                total_loss += loss.item()

            losses[language] = round(total_loss / num_batches, 4)

    losses["mean"] = round((losses["en"] + losses["tr"] + losses["zh"]) / 3, 4)
    model.train()                      # dropout back on
    return losses

def train(tokenizer_name, vocab_size, steps=6000, batch_size=64, context=256,
          learning_rate=3e-4, eval_every=200, seed=42):
    """Train one model for a fixed number of optimiser updates.
    """
    # fixed random seeds so the run can be reproduced 
    torch.manual_seed(seed)               
    sampler = np.random.default_rng(seed)  # which windows we sample

    run_dir = RESULTS_DIR / tokenizer_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = (run_dir/"log.jsonl").open("w")

    # the pre-tokenized id streams, cached on disk by dataset.py
    train_streams = load_streams(tokenizer_name, "train")
    valid_streams = load_streams(tokenizer_name, "valid")

    # build the model and move its weights onto the GPU
    model = TransformerLM(vocab_size, context=context).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"{tokenizer_name} on {DEVICE}, {steps} steps")
    best_val_loss = float("inf")     # start infinitely bad so the first result wins

    for step in range(1, steps + 1):
        inputs, targets = get_batch(train_streams, sampler, batch_size,
                                    context, DEVICE)

        predictions = model(inputs)
        loss = cross_entropy(predictions.reshape(-1, vocab_size),
                             targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # every eval_every steps, check validation loss and record a data point
        if step % eval_every == 0:
            val_losses = evaluate(model, valid_streams, sampler,
                                  batch_size=batch_size, context=context)

            # one JSON object per line: survives a crash, and can be read back
            # to plot the learning curves without retraining
            log_entry = {"step": step,
                         "train_loss": round(loss.item(), 4),
                         "val_en": val_losses["en"],
                         "val_tr": val_losses["tr"],
                         "val_zh": val_losses["zh"],
                         "val_mean": val_losses["mean"]}
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()    # write to disk now, so a killed run keeps its log

            print(f"  {step:>5}  train {log_entry['train_loss']:.3f}"
                  f"  val {val_losses['mean']:.3f}"
                  f"  (en {val_losses['en']:.2f}"
                  f" tr {val_losses['tr']:.2f}"
                  f" zh {val_losses['zh']:.2f})")

            # keep the weights from whenever validation loss was lowest --
            # this is the checkpoint the final test evaluation uses
            if val_losses["mean"] < best_val_loss:
                best_val_loss = val_losses["mean"]
                torch.save({"model": model.state_dict(),   # the learned weights
                            "vocab_size": vocab_size,      # stored so the file
                            "context": context,            # is self-describing
                            "step": step}, run_dir / "best.pt")

    log_file.close()
    print(f"  best {best_val_loss:.4f}\n")


if __name__ == "__main__":
    # the three tokenization conditions: same architecture, same step count,
    # only the vocabulary differs
    for tokenizer_name, vocab_size in [("char", 9702),
                                       ("bpe_2k", 2500),
                                       ("bpe_10k", 10000)]:
        train(tokenizer_name, vocab_size)
