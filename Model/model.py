"""
Encoder model definitions for the SEMSPEM pipeline.

- BiEncoder: bi-encoder used during WSD training; encodes both the target-word context and the WordNet gloss.
- SpanContextEncoder: thin wrapper around a tokenizer + transformer encoder,
  used during sparse indexing to embed passage token spans.
"""

from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BiEncoder(nn.Module):
    """A single shared transformer used to encode both contexts and glosses."""

    def __init__(self, pretrained: str, device):
        super().__init__()
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
        self.encoder   = AutoModel.from_pretrained(pretrained).to(device)

    def encode_context(self, tokens: List[str], span: tuple) -> torch.Tensor:
        """Mean-pool the sub-word representations covering the target span."""
        enc = self.tokenizer(
            tokens, is_split_into_words=True,
            return_tensors="pt", truncation=True
        ).to(self.device)
        with torch.set_grad_enabled(self.training):
            last_h = self.encoder(**enc).last_hidden_state[0]
        word_ids = enc.word_ids(0)
        idxs = [j for j, w in enumerate(word_ids)
                if w is not None and span[0] <= w < span[1]]
        if not idxs:
            idxs = [0]
        return last_h[idxs].mean(dim=0)

    def encode_gloss(self, gloss_text: str) -> torch.Tensor:
        """Encode a WordNet gloss string using its [CLS] representation."""
        enc = self.tokenizer(
            gloss_text, return_tensors="pt",
            truncation=True, max_length=128
        ).to(self.device)
        with torch.set_grad_enabled(self.training):
            cls_emb = self.encoder(**enc).last_hidden_state[0, 0]
        return cls_emb


class SpanContextEncoder(nn.Module):
    """Holds the tokenizer + transformer encoder used to embed passage spans during indexing."""

    def __init__(self, pretrained_model_name: str = "bert-base-uncased", device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name).to(self.device)
