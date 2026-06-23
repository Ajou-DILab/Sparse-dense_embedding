"""
Dataset and collate-function definitions for the SEMSPEM pipeline.

- BiEncoderWSDataset / wsd_collate_fn: training data for the WSD bi-encoder
  (SemCor context + gold synset + sampled negatives).
- MSMarcoTSVDataset / passage_collate_fn: passages to be indexed for sparse
  retrieval.
"""

import csv

import pandas as pd
from torch.utils.data import Dataset

from utils import clean_text_for_indexing, sample_negatives, tokenize_to_words


class BiEncoderWSDataset(Dataset):
    """SemCor-derived (context, gold synset) pairs with on-the-fly negative sampling."""

    def __init__(self, csv_path: str, synset2gloss, supersense2syns, lemma2syns,
                 n_easy: int, n_semi: int, n_hard: int):
        df = pd.read_csv(csv_path)

        self.contexts = df["context_tokens"].apply(eval).tolist()
        self.spans    = df["target_span"].apply(eval).tolist()
        self.labels   = df["synset_id"].tolist()

        valid_mask    = [sid in synset2gloss for sid in self.labels]
        self.contexts = [c for c, v in zip(self.contexts, valid_mask) if v]
        self.spans    = [s for s, v in zip(self.spans,    valid_mask) if v]
        self.labels   = [l for l, v in zip(self.labels,   valid_mask) if v]

        self.synset2gloss    = synset2gloss
        self.supersense2syns = supersense2syns
        self.lemma2syns      = lemma2syns
        self.n_easy, self.n_semi, self.n_hard = n_easy, n_semi, n_hard

        print(f"  Dataset size: {len(self.labels)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gold_sid = self.labels[idx]
        easy_negs, semi_negs, hard_negs = sample_negatives(
            gold_sid, self.supersense2syns, self.lemma2syns, self.synset2gloss,
            self.n_easy, self.n_semi, self.n_hard,
        )
        valid = (len(easy_negs) == self.n_easy and
                 len(semi_negs) == self.n_semi and
                 len(hard_negs) == self.n_hard)

        return {
            "tokens":    self.contexts[idx],
            "span":      self.spans[idx],
            "gold_sid":  gold_sid,
            "easy_negs": easy_negs,
            "semi_negs": semi_negs,
            "hard_negs": hard_negs,
            "valid":     valid,
        }


def wsd_collate_fn(batch):
    """Drop any sample where negative sampling could not fill the required quota."""
    return [b for b in batch if b["valid"]]


class MSMarcoTSVDataset(Dataset):
    """Loads (pid, text) pairs from an MS MARCO-style collection.tsv file."""

    def __init__(self, tsv_path: str, sample_limit: int = None):
        self.data = []
        print(f"Loading data from {tsv_path}...")
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if sample_limit and i >= sample_limit:
                    break
                pid = row[0].strip() if len(row) > 0 else ""
                if not pid:
                    pid = str(i)
                text = row[1].strip() if len(row) >= 2 else ""
                if not text:
                    continue
                self.data.append({'pid': pid, 'text': text})
        print(f"Loaded {len(self.data)} records.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def passage_collate_fn(batch, tokenizer_nlp):
    """Clean and tokenize a batch of raw passages."""
    pids = [b['pid'] for b in batch]
    raw_texts = [b['text'] for b in batch]
    clean_texts = [clean_text_for_indexing(t) for t in raw_texts]
    words = [tokenize_to_words(t, tokenizer_nlp) for t in clean_texts]
    return pids, words, clean_texts, raw_texts
