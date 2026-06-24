"""
Shared utility functions for the SEMSPEM pipeline.

This module collects WordNet-related helpers used during bi-encoder training
(sense/gloss bookkeeping, negative sampling) as well as text-processing and
indexing helpers used when building the sparse index (tokenization, POS
tagging, named-entity span alignment, medoid loading, SQLite setup).
"""

import io
import os
import re
import random
import zlib
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import joblib
import nltk
import torch
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

LEMM = WordNetLemmatizer()


# =============================================================================
# NLTK data setup (shared across entry scripts)
# =============================================================================
def ensure_nltk(nltk_dir: str = "./.nltk_data") -> None:
    """
    Make sure the NLTK corpora required by this project (wordnet, punkt,
    omw-1.4, stopwords, averaged_perceptron_tagger) are available, downloading
    them into `nltk_dir` if they're missing.
    """
    os.makedirs(nltk_dir, exist_ok=True)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    need = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
    ]
    for res_path, pkg in need:
        try:
            nltk.data.find(res_path)
        except LookupError:
            nltk.download(pkg, download_dir=nltk_dir, quiet=True)


# =============================================================================
# WordNet sense/gloss structures (used for bi-encoder training)
# =============================================================================
def build_wordnet_index() -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
    """
    Build the WordNet lookup tables used for negative sampling and gloss
    encoding during training.

    Gloss format: "target word: definition"
    e.g. bank.n.01 -> "bank: a financial institution that accepts deposits"

    Returns:
        supersense2syns: lexname (supersense) -> list of synset ids
        lemma2syns:       lemma -> list of synset ids
        synset2gloss:     synset id -> "target word: definition" string
    """
    print("Building WordNet index...")

    supersense2syns: Dict[str, List[str]] = defaultdict(list)
    lemma2syns:      Dict[str, List[str]] = defaultdict(list)
    synset2gloss:    Dict[str, str]       = {}

    for syn in wn.all_synsets():
        sid        = syn.name()
        supersense = syn.lexname()

        # e.g. "bank: a financial institution that accepts deposits"
        lemma_word = sid.split('.')[0].replace('_', ' ')  # e.g. "bank", "play on"
        definition = syn.definition()
        gloss      = f"{lemma_word}: {definition}"

        synset2gloss[sid] = gloss
        supersense2syns[supersense].append(sid)

        for lemma in syn.lemmas():
            lemma2syns[lemma.name()].append(sid)

    print(f"  Supersenses: {len(supersense2syns)}")
    print(f"  Synsets:     {len(synset2gloss)}")
    return supersense2syns, lemma2syns, synset2gloss


def sample_negatives(
    gold_synset_id: str,
    supersense2syns: Dict[str, List[str]],
    lemma2syns: Dict[str, List[str]],
    synset2gloss: Dict[str, str],
    n_easy: int,
    n_semi: int,
    n_hard: int,
):
    """
    Sample easy / semi-hard / hard negative synsets for a gold synset.

    - Hard:  other senses of the same lemma
    - Semi:  other synsets within the same supersense (lexname), excluding hard negatives
    - Easy:  synsets from a different supersense entirely
    """
    syn        = wn.synset(gold_synset_id)
    supersense = syn.lexname()
    lemma_name = gold_synset_id.split('.')[0]

    hard_pool = [s for s in lemma2syns.get(lemma_name, [])
                 if s != gold_synset_id and s in synset2gloss]

    semi_pool = [s for s in supersense2syns.get(supersense, [])
                 if s != gold_synset_id
                 and s not in hard_pool
                 and s in synset2gloss]

    easy_pool = [s for ss, syns in supersense2syns.items()
                 if ss != supersense
                 for s in syns
                 if s in synset2gloss]

    def safe_sample(pool, k):
        if len(pool) >= k:
            return random.sample(pool, k)
        return random.choices(pool, k=k) if pool else []

    return (safe_sample(easy_pool, n_easy),
            safe_sample(semi_pool, n_semi),
            safe_sample(hard_pool, n_hard))


# =============================================================================
# WordNet lemma index (lightweight version used during indexing)
# =============================================================================
def build_lemma_to_synsets() -> Dict[str, List[str]]:
    """Build a lemma -> [synset_id, ...] lookup over all of WordNet."""
    lemma2syns: Dict[str, List[str]] = defaultdict(list)
    for syn in wn.all_synsets():
        for lemma in syn.lemmas():
            lemma2syns[lemma.name().lower()].append(syn.name())
    return lemma2syns


# =============================================================================
# POS mapping & tokenization (used during indexing)
# =============================================================================
def to_wn_pos(ptb_tag: str) -> Optional[str]:
    """Map a Penn Treebank POS tag to the corresponding WordNet POS code."""
    if not ptb_tag:
        return None
    t = ptb_tag[0]
    if t == 'N':
        return wn.NOUN
    if t == 'V':
        return wn.VERB
    if t == 'J':
        return wn.ADJ
    if t == 'R':
        return wn.ADV
    return None


def clean_text_for_indexing(text: str) -> str:
    """Strip control characters and non-alphanumeric characters, collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_to_words(text: str, tokenizer_nlp) -> List[str]:
    """Tokenize cleaned text into words using a blank spaCy tokenizer."""
    clean_text = text.strip()
    if not clean_text:
        return []
    return [t.text for t in tokenizer_nlp.tokenizer(clean_text)]


def lemmatize_tokens(words: List[str]) -> Tuple[List[str], List[str]]:
    """POS-tag and lemmatize a list of tokens, returning (lemmas, pos_tags)."""
    tagged = pos_tag(words)
    lemmas, tags = [], []
    for w, tag in tagged:
        wn_pos = to_wn_pos(tag)
        w_low = w.lower()
        if wn_pos:
            lemmas.append(LEMM.lemmatize(w_low, pos=wn_pos))
        else:
            lemmas.append(LEMM.lemmatize(w_low))
        tags.append(tag)
    return lemmas, tags


# =============================================================================
# Named-entity span alignment (DBpedia char offsets -> word indices)
# =============================================================================
def build_char_to_word_idx(words: List[str], text: str) -> Dict[int, int]:
    """
    Map each token's character span (within `text`) to its word index.
    Returns {char_offset: word_idx}.
    """
    char_to_widx = {}
    pos = 0
    for widx, w in enumerate(words):
        found = text.find(w, pos)
        if found == -1:
            continue
        for c in range(found, found + len(w)):
            char_to_widx[c] = widx
        pos = found + len(w)
    return char_to_widx


def get_ne_word_indices(ent_start_char: int, ent_end_char: int,
                         char_to_widx: Dict[int, int]) -> set:
    """Return the set of word indices covered by an entity's character range."""
    indices = set()
    for c in range(ent_start_char, ent_end_char):
        if c in char_to_widx:
            indices.add(char_to_widx[c])
    return indices


# =============================================================================
# Medoid (WSI cluster) loading
# =============================================================================
def load_medoids(medoids_path: str, device) -> Optional[torch.Tensor]:
    """
    Load the medoid tensor produced by clustering the full (POS-agnostic)
    vocabulary. The file may be zlib-compressed; falls back to raw bytes if not.
    """
    print("\nLoading WSI Medoids ...")

    if not os.path.exists(medoids_path):
        print(f"  [ERROR] Medoids file not found: {medoids_path}")
        return None

    with open(medoids_path, 'rb') as f:
        file_content = f.read()

    try:
        raw_data = zlib.decompress(file_content)
    except Exception:
        raw_data = file_content

    try:
        data_dict = joblib.load(io.BytesIO(raw_data))
        keys = list(data_dict.keys())
        if 'medoids' in keys:
            medoids_array = data_dict['medoids']
            actual_k = len(medoids_array)
        else:
            actual_k = keys[0]
            medoids_array = data_dict[actual_k]['medoids']

        medoids_tensor = torch.from_numpy(medoids_array).float().to(device)
        print(f"  - Loaded global medoids!")
        print("WSI Medoids loading complete.\n")
        return medoids_tensor
    except Exception as e:
        print(f"  [ERROR] Error loading medoids: {e}")
        return None


# =============================================================================
# SQLite index schema
# =============================================================================
def init_db(db_path: str, store_doc_terms: bool = False):
    """Create (or recreate) the SQLite database with the index schema.

    Args:
        db_path: output SQLite path.
        store_doc_terms: if True, also create the optional `doc_terms` and
            `doc_text` tables. These store, per passage, the mapped term->tf
            dictionary and the raw passage text. They are not needed for
            scoring (the postings list is sufficient), but they let the
            retrieval script reverse-look-up *why* a document was retrieved
            (its indexed senses/entities) for inspection / qualitative output.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.executescript("""
        CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE doclen (pid TEXT PRIMARY KEY, dl INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE postings ( term TEXT PRIMARY KEY, df INTEGER NOT NULL, blob BLOB NOT NULL ) WITHOUT ROWID;
        CREATE TABLE tmp_tf ( term TEXT NOT NULL, pid TEXT NOT NULL, tf INTEGER NOT NULL );
        CREATE INDEX idx_tmp_tf_term ON tmp_tf(term);
        CREATE INDEX idx_tmp_tf_pid ON tmp_tf(pid);
        CREATE TABLE passage_confidence (
            pid              TEXT PRIMARY KEY,
            ne_conf_avg      REAL,
            ne_conf_cnt      INTEGER,
            wsd_wsi_conf_avg REAL,
            wsd_wsi_conf_cnt INTEGER
        ) WITHOUT ROWID;
    """)
    if store_doc_terms:
        cur.executescript("""
            CREATE TABLE doc_terms ( pid TEXT PRIMARY KEY, blob BLOB NOT NULL ) WITHOUT ROWID;
            CREATE TABLE doc_text  ( pid TEXT PRIMARY KEY, text TEXT ) WITHOUT ROWID;
        """)
    conn.commit()
    return conn
