# Assignment 1: Tokenization

Three tokenizers (char, BPE 2.5k, BPE 10k) and one small language model for
each, tested on English, Turkish and Chinese.

## How to run

On mltgpu, from this folder:

```
export DATA_ROOT=/srv/data/lt2326-h26/a1

python -m src.tokenizers.train_tokenizer      # train the tokenizers
python -m src.analysis.token_stats            # tokenizer table
python -m src.train                           # train the models (GPU)
python -m src.analysis.plots                  # learning curves
python -m src.evaluate                        # test BPC
python -m src.analysis.token_languages        # who gets the vocabulary
python -m src.analysis.error_analysis         # error analysis
```

Results are saved in `results/` and the figure in `figures/`.

## Trained models

`results/runs/<tokenizer>/best.pt`
