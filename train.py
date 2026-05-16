"""
Training, decoding, BLEU evaluation, and checkpoint utilities.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EOS_IDX, PAD_IDX, SOS_IDX, build_datasets, collate_batch
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        non_pad = target != self.pad_idx
        if non_pad.sum() == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            smooth_value = self.smoothing / max(self.vocab_size - 2, 1)
            true_dist.fill_(smooth_value)
            true_dist[:, self.pad_idx] = 0.0
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist.masked_fill_(~non_pad.unsqueeze(1), 0.0)

        loss = -(true_dist * log_probs).sum(dim=-1)
        return loss[non_pad].mean()


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(data_iter, desc=f"{'train' if is_train else 'eval'} epoch {epoch_num}", leave=False)

    for src, tgt in iterator:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_mask = make_src_mask(src, PAD_IDX)
        tgt_mask = make_tgt_mask(tgt_in, PAD_IDX)

        if is_train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if is_train and optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * max(tokens, 1)
        total_tokens += tokens
        iterator.set_postfix(loss=loss.item())

    return total_loss / max(total_tokens, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = torch.argmax(logits[:, -1, :], dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)
            if next_word == end_symbol:
                break
    return ys


def _tokens_from_vocab(vocab, ids):
    if hasattr(vocab, "decode"):
        return vocab.decode(ids, remove_specials=True)
    tokens = []
    for idx in ids:
        idx = int(idx)
        if hasattr(vocab, "lookup_token"):
            token = vocab.lookup_token(idx)
        elif hasattr(vocab, "itos"):
            token = vocab.itos[idx]
        else:
            token = str(idx)
        if token == "<eos>":
            break
        if token not in {"<unk>", "<pad>", "<sos>", "<eos>"}:
            tokens.append(token)
    return tokens


def _corpus_bleu(references: list[list[str]], hypotheses: list[list[str]], max_n: int = 4) -> float:
    weights = [1.0 / max_n] * max_n
    precisions = []
    hyp_len = 0
    ref_len = 0

    for n in range(1, max_n + 1):
        clipped = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = Counter(tuple(ref[i : i + n]) for i in range(max(len(ref) - n + 1, 0)))
            hyp_counts = Counter(tuple(hyp[i : i + n]) for i in range(max(len(hyp) - n + 1, 0)))
            clipped += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
            total += sum(hyp_counts.values())
            if n == 1:
                hyp_len += len(hyp)
                ref_len += len(ref)
        precisions.append((clipped + 1.0) / (total + 1.0))

    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
    score = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
    return 100.0 * score


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    references, hypotheses = [], []
    for src, tgt in tqdm(test_dataloader, desc="bleu", leave=False):
        for i in range(src.size(0)):
            src_i = src[i : i + 1].to(device)
            src_mask = make_src_mask(src_i, PAD_IDX)
            pred = greedy_decode(model, src_i, src_mask, max_len, SOS_IDX, EOS_IDX, device=device)
            hypotheses.append(_tokens_from_vocab(tgt_vocab, pred.squeeze(0).tolist()))
            references.append(_tokens_from_vocab(tgt_vocab, tgt[i].tolist()))
    return _corpus_bleu(references, hypotheses)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model.model_config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", 0))


def run_training_experiment() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds, src_vocab, tgt_vocab = build_datasets(
        args.min_freq,
        args.max_vocab_size,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": lambda batch: collate_batch(batch, PAD_IDX),
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = Transformer(
        len(src_vocab),
        len(tgt_vocab),
        d_model=args.d_model,
        N=args.layers,
        num_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, args.label_smoothing)

    wandb = None
    if args.wandb:
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(project="da6401-a3", config=vars(args))

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, device)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, args.checkpoint)
        if wandb is not None:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    if wandb is not None:
        wandb.log({"test_bleu": bleu})
    print(f"test_bleu={bleu:.2f}")


if __name__ == "__main__":
    run_training_experiment()
