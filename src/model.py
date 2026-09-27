import csv
from pathlib import Path

import torch
import torch.nn as nn

OUT_PATH = Path("results/parameter_counts.csv")

# The three tokenization conditions. 
CONDITIONS = {
    "char":    9702,
    "bpe_2k":  2500,
    "bpe_10k": 10000,
}

class TransformerLM(nn.Module):
    """Small decoder-only Transformer language model."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=2,
                 d_ff=1024, dropout=0.1, context=256):
        super().__init__()
        self.context = context
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        h = self.drop(self.token_emb(x) + self.pos_emb(positions))
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


def parameter_counts(vocab_size):
    """The three numbers the assignment asks for, plus the shared body."""
    model = TransformerLM(vocab_size)
    total = sum(p.numel() for p in model.parameters())
    input_emb = model.token_emb.weight.numel()
    output_layer = sum(p.numel() for p in model.head.parameters())
    return {
        "vocab_size": vocab_size,
        "total": total,
        "input_emb": input_emb,
        "output_layer": output_layer,
        "body": total - input_emb - output_layer,
    }


def check_causal():
    """A token must not affect the outputs at earlier positions."""
    torch.manual_seed(0)
    model = TransformerLM(100, context=8).eval()
    x = torch.randint(0, 100, (1, 8))
    x2 = x.clone()
    x2[0, 4] += 1

    with torch.no_grad():
        a, b = model(x), model(x2)

    assert torch.allclose(a[0, :4], b[0, :4], atol=1e-6), "mask leaks: earlier positions changed"
    assert not torch.allclose(a[0, 4:], b[0, 4:], atol=1e-6), "model ignores its input"
    print("causal mask OK")


if __name__ == "__main__":
    check_causal()

    rows = [{"condition": name, **parameter_counts(v)} for name, v in CONDITIONS.items()]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_PATH}")

    for r in rows:
        print(f"  {r['condition']:<9} vocab {r['vocab_size']:>6}  "
              f"total {r['total']:>10,}  in_emb {r['input_emb']:>9,}  "
              f"out {r['output_layer']:>9,}  body {r['body']:>9,}")