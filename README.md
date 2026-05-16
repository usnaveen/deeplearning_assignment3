# DA6401 Assignment 3 - Transformer for German to English Translation

This repository implements the Transformer architecture from "Attention Is All You Need" in PyTorch for Multi30k German to English translation.

## Files

- `model.py`: scaled dot-product attention, multi-head attention, masks, sinusoidal positional encoding, encoder/decoder stacks, full `Transformer`.
- `lr_scheduler.py`: Noam learning-rate scheduler with warmup and inverse-square-root decay.
- `dataset.py`: Multi30k loading from Hugging Face, spaCy tokenization, vocabulary construction, and padded batching.
- `train.py`: label smoothing, training/evaluation loop, greedy decoding, corpus BLEU, checkpoint save/load, CLI entry point.

## Install

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

If the spaCy language models are unavailable, `dataset.py` falls back to blank spaCy tokenizers.

## Train

```bash
python train.py \
  --epochs 10 \
  --batch-size 64 \
  --d-model 256 \
  --layers 3 \
  --heads 8 \
  --d-ff 512 \
  --warmup-steps 4000 \
  --label-smoothing 0.1 \
  --checkpoint checkpoint.pt
```

Enable Weights & Biases logging with:

```bash
python train.py --wandb
```

For a quick local smoke test on an 8GB laptop:

```bash
python train.py \
  --epochs 1 \
  --batch-size 8 \
  --d-model 64 \
  --layers 1 \
  --heads 4 \
  --d-ff 128 \
  --train-limit 256 \
  --val-limit 64 \
  --test-limit 16 \
  --max-vocab-size 2000
```

## Design Note

The encoder and decoder use pre-layer normalization. This keeps the assignment's required attention, feed-forward, residual, and normalization structure, while improving optimization stability for smaller training runs.

## Autograder Contract

The public names from the official skeleton are preserved:

- `scaled_dot_product_attention`
- `MultiHeadAttention`
- `PositionalEncoding`
- `make_src_mask`
- `make_tgt_mask`
- `Transformer.encode`
- `Transformer.decode`
- `LabelSmoothingLoss`
- `greedy_decode`
- `evaluate_bleu`
- `save_checkpoint`
- `load_checkpoint`
- `NoamScheduler`
