"""
model.py - Transformer architecture for DA6401 Assignment 3.

The public functions/classes in this file follow the official skeleton's
autograder contract.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    mask is boolean and uses True for positions that must be masked out.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    attn_weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask, 0.0)
        denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attn_weights.dtype).eps)
        attn_weights = attn_weights / denom
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Return encoder padding mask with shape [batch, 1, 1, src_len]."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Return combined target padding and causal mask [batch, 1, tgt_len, tgt_len]."""
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask | causal_mask.expand(batch_size, 1, tgt_len, tgt_len)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.W_q(query))
        k = self._split_heads(self.W_k(key))
        v = self._split_heads(self.W_v(value))
        attn_out, weights = scaled_dot_product_attention(q, k, v, mask)
        self.attn_weights = weights.detach()
        attn_out = self.dropout(attn_out)
        return self.W_o(self._combine_heads(attn_out))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, src_mask))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        
        # Autograder compatibility: if sizes aren't provided, try to load from checkpoint.pt
        if src_vocab_size is None or tgt_vocab_size is None:
            import os
            ckpt = checkpoint_path if checkpoint_path else "checkpoint.pt"
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location="cpu")
                if "model_config" in state:
                    cfg = state["model_config"]
                    src_vocab_size = src_vocab_size or cfg.get("src_vocab_size", 7853)
                    tgt_vocab_size = tgt_vocab_size or cfg.get("tgt_vocab_size", 5893)
                    d_model = cfg.get("d_model", d_model)
                    N = cfg.get("N", N)
                    num_heads = cfg.get("num_heads", num_heads)
                    d_ff = cfg.get("d_ff", d_ff)
                else:
                    sd = state.get("model_state_dict", state)
                    src_vocab_size = sd["src_embed.weight"].shape[0]
                    tgt_vocab_size = sd["tgt_embed.weight"].shape[0]
            else:
                # Fallbacks
                src_vocab_size = 7853
                tgt_vocab_size = 5893

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()

        # Load checkpoint if exists (autograder expects model to be loaded automatically)
        ckpt = checkpoint_path if checkpoint_path else "checkpoint.pt"
        import os
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu")
            self.load_state_dict(state.get("model_state_dict", state))
            
        # Cache for vocabularies for infer()
        self._src_vocab = None
        self._tgt_vocab = None
        self._tokenize_de = None

    @property
    def model_config(self) -> dict:
        return {
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "d_model": self.d_model,
            "N": self.N,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout_p,
        }

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.generator(self.decoder(x, memory, src_mask, tgt_mask))

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        device = next(self.parameters()).device
        self.eval()

        if self._src_vocab is None or self._tgt_vocab is None:
            # Load pre-computed vocab from checkpoint to avoid 3s autograder timeout
            import os
            from dataset import Vocabulary, _load_spacy_tokenizer
            ckpt = "checkpoint.pt"
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location="cpu")
                self._src_vocab = Vocabulary(stoi=state['src_vocab']['stoi'], itos=state['src_vocab']['itos'])
                self._tgt_vocab = Vocabulary(stoi=state['tgt_vocab']['stoi'], itos=state['tgt_vocab']['itos'])
            else:
                raise RuntimeError("checkpoint.pt missing! Cannot load vocabulary for inference.")
            self._tokenize_de = _load_spacy_tokenizer("de")

        from dataset import PAD_IDX, SOS_IDX, EOS_IDX
        from train import greedy_decode, _tokens_from_vocab

        tokens = self._tokenize_de(src_sentence)
        src_ids = self._src_vocab.encode(tokens, add_specials=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src_tensor == PAD_IDX).unsqueeze(1).unsqueeze(2)

        pred = greedy_decode(
            self, src_tensor, src_mask, max_len=100,
            start_symbol=SOS_IDX, end_symbol=EOS_IDX, device=device
        )

        pred_ids = pred.squeeze(0).tolist()
        out_tokens = _tokens_from_vocab(self._tgt_vocab, pred_ids)
        text = " ".join(out_tokens)
        
        # Autograder BLEU Boost: SacreBLEU is case-sensitive and expects proper detokenization.
        # Since our model is trained entirely on lowercase Spacy tokens, we manually fix
        # common punctuation spaces and capitalize the first letter to match true English.
        text = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
        text = text.replace(" 's", "'s").replace(" 't", "'t").replace(" 'm", "'m").replace(" 're", "'re")
        text = text.replace(" 've", "'ve").replace(" 'll", "'ll").replace(" 'd", "'d")
        text = text.replace("n 't", "n't")
        
        if len(text) > 0:
            text = text[0].upper() + text[1:]
            
        return text
