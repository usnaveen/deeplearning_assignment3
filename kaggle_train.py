"""
DA6401 Assignment 3 - Full Training + W&B Experiments for Kaggle.
Run on a Kaggle GPU notebook.

Usage in Kaggle notebook cell:
    !pip install torch numpy matplotlib scikit-learn wandb datasets spacy tqdm
    !python -m spacy download de_core_news_sm
    !python -m spacy download en_core_web_sm
    !git clone https://github.com/usnaveen/deeplearning_assignment3.git
    %cd deeplearning_assignment3
    !python kaggle_train.py --wandb-key YOUR_WANDB_KEY
"""

from __future__ import annotations
import argparse, copy, math, os, json
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EOS_IDX, PAD_IDX, SOS_IDX, build_datasets, collate_batch
from lr_scheduler import NoamScheduler
from model import (
    Transformer, MultiHeadAttention, PositionalEncoding,
    make_src_mask, make_tgt_mask, scaled_dot_product_attention,
)
from train import (
    LabelSmoothingLoss, run_epoch, greedy_decode,
    evaluate_bleu, save_checkpoint, load_checkpoint, _tokens_from_vocab,
)

# ---------------------------------------------------------------------------
# Learned Positional Encoding variant for Experiment 2.4
# ---------------------------------------------------------------------------
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


# ---------------------------------------------------------------------------
# Unscaled attention for Experiment 2.2
# ---------------------------------------------------------------------------
def unscaled_dot_product_attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1))  # NO /sqrt(dk)
    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    attn_weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask, 0.0)
        denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attn_weights = attn_weights / denom
    return torch.matmul(attn_weights, V), attn_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_model(src_vocab_size, tgt_vocab_size, cfg, device, use_learned_pe=False, use_unscaled=False):
    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model=cfg["d_model"], N=cfg["layers"],
        num_heads=cfg["heads"], d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    ).to(device)

    if use_learned_pe:
        model.positional_encoding = LearnedPositionalEncoding(
            cfg["d_model"], cfg["dropout"]
        ).to(device)

    if use_unscaled:
        import model as model_module
        model_module.scaled_dot_product_attention = unscaled_dot_product_attention

    return model


def get_dataloaders(cfg):
    train_ds, val_ds, test_ds, src_vocab, tgt_vocab = build_datasets(
        cfg["min_freq"], cfg.get("max_vocab_size"),
        train_limit=cfg.get("train_limit"),
        val_limit=cfg.get("val_limit"),
        test_limit=cfg.get("test_limit"),
    )
    kw = {"batch_size": cfg["batch_size"], "collate_fn": lambda b: collate_batch(b, PAD_IDX), "num_workers": 2}
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        src_vocab, tgt_vocab,
    )


def train_model(model, train_loader, val_loader, test_loader, tgt_vocab,
                loss_fn, optimizer, scheduler, cfg, device, wandb_run, tag=""):
    best_val = float("inf")
    ckpt_path = f"checkpoint_{tag}.pt" if tag else "checkpoint.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, device)

        log = {"epoch": epoch, f"{tag}/train_loss": train_loss, f"{tag}/val_loss": val_loss}

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path)

        # Compute val BLEU every 3 epochs
        if epoch % 3 == 0 or epoch == cfg["epochs"]:
            bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device, max_len=100)
            log[f"{tag}/val_bleu"] = bleu
            print(f"[{tag}] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_bleu={bleu:.2f}")
        else:
            print(f"[{tag}] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if wandb_run:
            wandb_run.log(log)

    # Final test BLEU
    load_checkpoint(ckpt_path, model)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=100)
    print(f"[{tag}] test_bleu={test_bleu:.2f}")
    if wandb_run:
        wandb_run.log({f"{tag}/test_bleu": test_bleu})
    return ckpt_path, test_bleu


# ===================== EXPERIMENTS =====================

def experiment_main(cfg, device, wandb_module):
    print("\n" + "="*60)
    print("MAIN TRAINING RUN")
    print("="*60)
    run = wandb_module.init(project="da6401-a3", name="main_noam", config=cfg, reinit=True)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(cfg)
    model = build_model(len(src_vocab), len(tgt_vocab), cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=0.1)

    ckpt, bleu = train_model(model, train_loader, val_loader, test_loader, tgt_vocab,
                             loss_fn, optimizer, scheduler, cfg, device, run, tag="main")
    run.finish()
    return ckpt


def experiment_2_1(cfg, device, wandb_module):
    print("\n" + "="*60)
    print("EXPERIMENT 2.1: Noam vs Fixed LR (with LR Tracking)")
    print("="*60)
    run = wandb_module.init(project="da6401-a3", name="exp2.1_noam_vs_fixed", config=cfg, reinit=True)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(cfg)

    model_a = build_model(len(src_vocab), len(tgt_vocab), cfg, device)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched_a = NoamScheduler(opt_a, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
    loss_a = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, 0.1)
    
    model_b = build_model(len(src_vocab), len(tgt_vocab), cfg, device)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    loss_b = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, 0.1)

    for ep in range(1, cfg["epochs"] + 1):
        tl_a = run_epoch(train_loader, model_a, loss_a, opt_a, sched_a, ep, True, device)
        vl_a = run_epoch(val_loader, model_a, loss_a, None, None, ep, False, device)
        lr_a = opt_a.param_groups[0]['lr']
        
        tl_b = run_epoch(train_loader, model_b, loss_b, opt_b, None, ep, True, device)
        vl_b = run_epoch(val_loader, model_b, loss_b, None, None, ep, False, device)
        lr_b = opt_b.param_groups[0]['lr']
        
        run.log({
            "epoch": ep,
            "noam/train_loss": tl_a, "noam/val_loss": vl_a, "noam/learning_rate": lr_a,
            "fixed/train_loss": tl_b, "fixed/val_loss": vl_b, "fixed/learning_rate": lr_b
        })
        print(f"[noam] ep={ep} vl={vl_a:.4f} lr={lr_a:.6f} | [fixed] vl={vl_b:.4f} lr={lr_b:.6f}")
    run.finish()


def experiment_2_2(cfg, device, wandb_module):
    print("\n" + "="*60)
    print("EXPERIMENT 2.2: Scaling Factor Ablation (with Entropy)")
    print("="*60)
    import model as model_module
    original_attn = model_module.scaled_dot_product_attention
    run = wandb_module.init(project="da6401-a3", name="exp2.2_scaling", config=cfg, reinit=True)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(cfg)

    for variant, attn_fn in [("scaled", original_attn), ("unscaled", unscaled_dot_product_attention)]:
        model_module.scaled_dot_product_attention = attn_fn
        model = build_model(len(src_vocab), len(tgt_vocab), cfg, device)
        opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamScheduler(opt, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, 0.1)

        step = 0
        for ep in range(1, min(cfg["epochs"], 5) + 1):
            model.train()
            for src, tgt in train_loader:
                if step >= 1000: break
                src, tgt = src.to(device), tgt.to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                opt.zero_grad(set_to_none=True)
                logits = model(src, tgt_in, make_src_mask(src, PAD_IDX), make_tgt_mask(tgt_in, PAD_IDX))
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                loss.backward()
                
                # Gradients & Entropy
                q_norms, k_norms = [], []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if "W_q" in name: q_norms.append(param.grad.norm().item())
                        elif "W_k" in name: k_norms.append(param.grad.norm().item())
                        
                attn_weights = model.encoder.layers[-1].self_attn.attn_weights
                # Entropy = -sum(p * log(p))
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1).mean().item()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                
                if step <= 1000:
                    run.log({
                        "step": step,
                        f"{variant}/loss": loss.item(),
                        f"{variant}/grad_norm_Q": np.mean(q_norms) if q_norms else 0,
                        f"{variant}/grad_norm_K": np.mean(k_norms) if k_norms else 0,
                        f"{variant}/attention_entropy": entropy
                    })
            if step >= 1000: break
        print(f"[{variant}] logged {step} steps")

    model_module.scaled_dot_product_attention = original_attn
    run.finish()


def save_heatmap(attn, src_tokens, tgt_tokens, filename, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    num_heads = attn.shape[0]
    fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(4 * ((num_heads + 1) // 2), 8))
    axes = axes.flatten()
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn[h], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h}")
        ax.set_xticks(range(len(tgt_tokens)))
        ax.set_yticks(range(len(src_tokens)))
        ax.set_xticklabels(tgt_tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(src_tokens, fontsize=6)
    for h in range(num_heads, len(axes)):
        axes[h].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def experiment_2_3(cfg, device, wandb_module, checkpoint_path):
    print("\n" + "="*60)
    print("EXPERIMENT 2.3: Layer Progression & Cross Attention Heatmaps")
    print("="*60)
    run = wandb_module.init(project="da6401-a3", name="exp2.3_attention", config=cfg, reinit=True)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(cfg)

    model = build_model(len(src_vocab), len(tgt_vocab), cfg, device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    src_batch, tgt_batch = next(iter(test_loader))
    src_i = src_batch[0:1].to(device)
    tgt_i = tgt_batch[0:1].to(device)
    tgt_in = tgt_i[:, :-1]
    
    src_mask = make_src_mask(src_i, PAD_IDX)
    tgt_mask = make_tgt_mask(tgt_in, PAD_IDX)

    with torch.no_grad():
        _ = model(src_i, tgt_in, src_mask, tgt_mask)

    src_tokens = src_vocab.decode(src_i.squeeze(0).tolist(), remove_specials=False)
    tgt_tokens = tgt_vocab.decode(tgt_in.squeeze(0).tolist(), remove_specials=False)
    
    # Encoder Layer 1 (Local)
    attn_enc_1 = model.encoder.layers[0].self_attn.attn_weights.squeeze(0).cpu().numpy()
    seq_len = min(len(src_tokens), attn_enc_1.shape[1])
    save_heatmap(attn_enc_1[:, :seq_len, :seq_len], src_tokens[:seq_len], src_tokens[:seq_len], 
                 "enc_layer1.png", "Encoder Layer 1 (Self Attention)")
                 
    # Encoder Layer 3 (Global)
    attn_enc_3 = model.encoder.layers[-1].self_attn.attn_weights.squeeze(0).cpu().numpy()
    save_heatmap(attn_enc_3[:, :seq_len, :seq_len], src_tokens[:seq_len], src_tokens[:seq_len], 
                 "enc_layer3.png", f"Encoder Layer {cfg['layers']} (Self Attention)")
                 
    # Decoder Cross Attention
    attn_cross = model.decoder.layers[-1].src_attn.attn_weights.squeeze(0).cpu().numpy()
    save_heatmap(attn_cross[:, :len(tgt_tokens), :seq_len], tgt_tokens, src_tokens[:seq_len], 
                 "cross_attn.png", "Decoder -> Encoder Cross Attention")

    run.log({
        "encoder_layer_1": wandb_module.Image("enc_layer1.png"),
        "encoder_layer_3": wandb_module.Image("enc_layer3.png"),
        "decoder_cross_attn": wandb_module.Image("cross_attn.png"),
    })
    print("Logged 3 advanced heatmaps!")
    run.finish()


def evaluate_bucketed_bleu(model, test_loader, tgt_vocab, device):
    from train import greedy_decode, _tokens_from_vocab, _corpus_bleu
    buckets = {"short (<10)": (0, 10), "medium (10-20)": (10, 20), "long (>20)": (20, 999)}
    refs = {k: [] for k in buckets}
    hyps = {k: [] for k in buckets}
    
    for src, tgt in test_loader:
        for i in range(src.size(0)):
            src_i = src[i:i+1].to(device)
            # Find true length without padding
            length = (src_i[0] != PAD_IDX).sum().item()
            
            bucket_name = None
            for name, (low, high) in buckets.items():
                if low <= length < high:
                    bucket_name = name
                    break
            if not bucket_name: continue
                
            src_mask = make_src_mask(src_i, PAD_IDX)
            pred = greedy_decode(model, src_i, src_mask, 100, SOS_IDX, EOS_IDX, device=device)
            hyps[bucket_name].append(_tokens_from_vocab(tgt_vocab, pred.squeeze(0).tolist()))
            refs[bucket_name].append(_tokens_from_vocab(tgt_vocab, tgt[i].tolist()))
            
    scores = {}
    for name in buckets:
        if len(refs[name]) > 0:
            scores[name] = _corpus_bleu(refs[name], hyps[name])
    return scores


def experiment_2_4(cfg, device, wandb_module):
    print("\n" + "="*60)
    print("EXPERIMENT 2.4: Bucketed BLEU Extrapolation")
    print("="*60)
    run = wandb_module.init(project="da6401-a3", name="exp2.4_pe", config=cfg, reinit=True)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(cfg)

    for variant, use_learned in [("sinusoidal", False), ("learned", True)]:
        model = build_model(len(src_vocab), len(tgt_vocab), cfg, device, use_learned_pe=use_learned)
        opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamScheduler(opt, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, 0.1)

        for ep in range(1, cfg["epochs"] + 1):
            tl = run_epoch(train_loader, model, loss_fn, opt, sched, ep, True, device)
            vl = run_epoch(val_loader, model, loss_fn, None, None, ep, False, device)
            log = {"epoch": ep, f"{variant}/train_loss": tl, f"{variant}/val_loss": vl}
            
            if ep == cfg["epochs"]:
                scores = evaluate_bucketed_bleu(model, test_loader, tgt_vocab, device)
                for bucket, score in scores.items():
                    log[f"{variant}/bleu_{bucket}"] = score
                    print(f"[{variant}] bucket {bucket} BLEU = {score:.2f}")
            print(f"[{variant}] ep={ep} tl={tl:.4f} vl={vl:.4f}")
            run.log(log)
    run.finish()


def experiment_2_5(cfg, device, wandb_module):
    print("\n" + "="*60)
    print("EXPERIMENT 2.5: Label Smoothing (with Output Entropy)")
    print("="*60)
    run = wandb_module.init(project="da6401-a3", name="exp2.5_label_smooth", config=cfg, reinit=True)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(cfg)

    for variant, eps in [("smooth_0.1", 0.1), ("smooth_0.0", 0.0)]:
        model = build_model(len(src_vocab), len(tgt_vocab), cfg, device)
        opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamScheduler(opt, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=eps)

        for ep in range(1, cfg["epochs"] + 1):
            model.train()
            total_conf, total_ent, total_tok = 0.0, 0.0, 0
            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                opt.zero_grad(set_to_none=True)
                logits = model(src, tgt_in, make_src_mask(src, PAD_IDX), make_tgt_mask(tgt_in, PAD_IDX))
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)
                    flat_probs = probs.reshape(-1, probs.size(-1))
                    flat_tgt = tgt_out.reshape(-1)
                    non_pad = flat_tgt != PAD_IDX
                    if non_pad.sum() > 0:
                        correct_probs = flat_probs[torch.arange(flat_probs.size(0), device=device), flat_tgt]
                        total_conf += correct_probs[non_pad].sum().item()
                        
                        # Output Entropy
                        ent = -torch.sum(flat_probs[non_pad] * torch.log(flat_probs[non_pad] + 1e-9), dim=-1)
                        total_ent += ent.sum().item()
                        total_tok += non_pad.sum().item()

            avg_conf = total_conf / max(total_tok, 1)
            avg_ent = total_ent / max(total_tok, 1)
            vl = run_epoch(val_loader, model, loss_fn, None, None, ep, False, device)
            
            run.log({"epoch": ep, f"{variant}/val_loss": vl, f"{variant}/pred_confidence": avg_conf, f"{variant}/output_entropy": avg_ent})
            print(f"[{variant}] ep={ep} vl={vl:.4f} conf={avg_conf:.4f} ent={avg_ent:.4f}")
    run.finish()


def experiment_bonus_1(cfg, device, wandb_module):
    print("\n" + "="*60)
    print("BONUS 1: Multi-Head Scaling")
    print("="*60)
    run = wandb_module.init(project="da6401-a3", name="bonus1_multi_head", config=cfg, reinit=True)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(cfg)

    for heads in [1, 4, 8]:
        variant = f"heads_{heads}"
        cfg_mod = cfg.copy()
        cfg_mod["heads"] = heads
        
        model = build_model(len(src_vocab), len(tgt_vocab), cfg_mod, device)
        opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamScheduler(opt, d_model=cfg_mod["d_model"], warmup_steps=cfg_mod["warmup_steps"])
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=0.1)

        for ep in range(1, 11): # Train for 10 epochs to save time
            tl = run_epoch(train_loader, model, loss_fn, opt, sched, ep, True, device)
            vl = run_epoch(val_loader, model, loss_fn, None, None, ep, False, device)
            log = {"epoch": ep, f"{variant}/train_loss": tl, f"{variant}/val_loss": vl}
            if ep == 10:
                bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device)
                log[f"{variant}/val_bleu"] = bleu
                print(f"[{variant}] ep={ep} val_bleu={bleu:.2f}")
            run.log(log)
    run.finish()


def experiment_bonus_2(cfg, device, wandb_module):
    print("\n" + "="*60)
    print("BONUS 2: Depth Scaling")
    print("="*60)
    run = wandb_module.init(project="da6401-a3", name="bonus2_depth", config=cfg, reinit=True)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(cfg)

    for layers in [1, 3, 6]:
        variant = f"layers_{layers}"
        cfg_mod = cfg.copy()
        cfg_mod["layers"] = layers
        
        model = build_model(len(src_vocab), len(tgt_vocab), cfg_mod, device)
        opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamScheduler(opt, d_model=cfg_mod["d_model"], warmup_steps=cfg_mod["warmup_steps"])
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=0.1)

        for ep in range(1, 11):
            tl = run_epoch(train_loader, model, loss_fn, opt, sched, ep, True, device)
            vl = run_epoch(val_loader, model, loss_fn, None, None, ep, False, device)
            log = {"epoch": ep, f"{variant}/train_loss": tl, f"{variant}/val_loss": vl}
            if ep == 10:
                bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device)
                log[f"{variant}/val_bleu"] = bleu
                print(f"[{variant}] ep={ep} val_bleu={bleu:.2f}")
            run.log(log)
    run.finish()


def main():
    import argparse, os, wandb
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    key = args.wandb_key
    if key is None: key = os.environ.get("WANDB_API_KEY")
    if key is None:
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except: pass
    if key: wandb.login(key=key)
    else: wandb.login()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "d_model": args.d_model, "layers": args.layers,
        "heads": args.heads, "d_ff": args.d_ff,
        "dropout": args.dropout, "warmup_steps": args.warmup_steps,
        "min_freq": args.min_freq, "max_vocab_size": None,
        "train_limit": None, "val_limit": None, "test_limit": None,
    }

    experiments = {
        "main": lambda: experiment_main(cfg, device, wandb),
        "2.1": lambda: experiment_2_1(cfg, device, wandb),
        "2.2": lambda: experiment_2_2(cfg, device, wandb),
        "2.3": lambda: experiment_2_3(cfg, device, wandb, "checkpoint.pt"),
        "2.4": lambda: experiment_2_4(cfg, device, wandb),
        "2.5": lambda: experiment_2_5(cfg, device, wandb),
        "bonus1": lambda: experiment_bonus_1(cfg, device, wandb),
        "bonus2": lambda: experiment_bonus_2(cfg, device, wandb),
    }

    keys = [k.strip() for k in args.only.split(",")] if args.only else list(experiments.keys())
    if args.skip_main and "main" in keys: keys.remove("main")

    for k in keys:
        if k in experiments: experiments[k]()
        else: print(f"Unknown: {k}")

if __name__ == "__main__":
    main()
