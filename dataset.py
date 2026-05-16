"""
Multi30k data loading, spaCy tokenization, vocabulary, and batching helpers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = range(4)


def _load_spacy_tokenizer(lang: str) -> Callable[[str], list[str]]:
    import spacy

    model_name = {"de": "de_core_news_sm", "en": "en_core_web_sm"}[lang]
    try:
        nlp = spacy.load(model_name)
    except OSError:
        nlp = spacy.blank(lang)
    return lambda text: [tok.text.lower() for tok in nlp.tokenizer(text)]


@dataclass
class Vocabulary:
    stoi: dict[str, int]
    itos: list[str]
    unk_idx: int = UNK_IDX
    pad_idx: int = PAD_IDX
    sos_idx: int = SOS_IDX
    eos_idx: int = EOS_IDX

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        min_freq: int = 2,
        max_size: int | None = None,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for tokens in token_sequences:
            counter.update(tokens)
        words = [word for word, freq in counter.items() if freq >= min_freq and word not in SPECIALS]
        words.sort(key=lambda word: (-counter[word], word))
        if max_size is not None:
            words = words[: max(0, max_size - len(SPECIALS))]
        itos = SPECIALS + words
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def encode(self, tokens: Sequence[str], add_specials: bool = True) -> list[int]:
        ids = [self[token] for token in tokens]
        if add_specials:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: Sequence[int], remove_specials: bool = True) -> list[str]:
        tokens = []
        for idx in ids:
            token = self.itos[int(idx)]
            if remove_specials and token in SPECIALS:
                if token == "<eos>":
                    break
                continue
            tokens.append(token)
        return tokens


def _load_hf_split(split: str):
    from datasets import load_dataset

    cache_dir = Path(__file__).resolve().parent / ".hf_cache"
    cache_dir.mkdir(exist_ok=True)
    return load_dataset("bentrevett/multi30k", split=split, cache_dir=str(cache_dir))


def _extract_pair(example) -> tuple[str, str]:
    if "translation" in example:
        return example["translation"]["de"], example["translation"]["en"]
    if "de" in example and "en" in example:
        return example["de"], example["en"]
    if "text_de" in example and "text_en" in example:
        return example["text_de"], example["text_en"]
    raise KeyError(f"Cannot find German/English fields in example keys: {list(example.keys())}")


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        src_vocab: Vocabulary | None = None,
        tgt_vocab: Vocabulary | None = None,
        min_freq: int = 2,
        max_vocab_size: int | None = None,
        limit: int | None = None,
    ) -> None:
        self.split = split
        self.raw_data = _load_hf_split(split)
        if limit is not None:
            self.raw_data = self.raw_data.select(range(min(limit, len(self.raw_data))))
        self.tokenize_de = _load_spacy_tokenizer("de")
        self.tokenize_en = _load_spacy_tokenizer("en")

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab(min_freq=min_freq, max_size=max_vocab_size)
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        self.examples = self.process_data()

    def build_vocab(self, min_freq: int = 2, max_size: int | None = None) -> tuple[Vocabulary, Vocabulary]:
        train_data = self.raw_data if self.split == "train" else _load_hf_split("train")
        de_tokens, en_tokens = [], []
        for example in train_data:
            de, en = _extract_pair(example)
            de_tokens.append(self.tokenize_de(de))
            en_tokens.append(self.tokenize_en(en))
        return (
            Vocabulary.build(de_tokens, min_freq=min_freq, max_size=max_size),
            Vocabulary.build(en_tokens, min_freq=min_freq, max_size=max_size),
        )

    def process_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        processed = []
        for example in self.raw_data:
            de, en = _extract_pair(example)
            src = torch.tensor(self.src_vocab.encode(self.tokenize_de(de)), dtype=torch.long)
            tgt = torch.tensor(self.tgt_vocab.encode(self.tokenize_en(en)), dtype=torch.long)
            processed.append((src, tgt))
        return processed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


def collate_batch(batch: Sequence[tuple[torch.Tensor, torch.Tensor]], pad_idx: int = PAD_IDX):
    src_batch, tgt_batch = zip(*batch)
    src = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src, tgt


def build_datasets(
    min_freq: int = 2,
    max_vocab_size: int | None = None,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
):
    train = Multi30kDataset(
        "train",
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        limit=train_limit,
    )
    val = Multi30kDataset("validation", src_vocab=train.src_vocab, tgt_vocab=train.tgt_vocab, limit=val_limit)
    test = Multi30kDataset("test", src_vocab=train.src_vocab, tgt_vocab=train.tgt_vocab, limit=test_limit)
    return train, val, test, train.src_vocab, train.tgt_vocab
