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



# NLTK data setup (shared across entry scripts)

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


def build_wordnet_index() -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
    print("Building WordNet index...")

    supersense2syns: Dict[str, List[str]] = defaultdict(list)
    lemma2syns:      Dict[str, List[str]] = defaultdict(list)
    synset2gloss:    Dict[str, str]       = {}

    for syn in wn.all_synsets():
        sid        = syn.name()
        supersense = syn.lexname()

        lemma_word = sid.split('.')[0].replace('_', ' ')  
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


def build_lemma_to_synsets() -> Dict[str, List[str]]:
    """Build a lemma -> [synset_id, ...] lookup over all of WordNet."""
    lemma2syns: Dict[str, List[str]] = defaultdict(list)
    for syn in wn.all_synsets():
        for lemma in syn.lemmas():
            lemma2syns[lemma.name().lower()].append(syn.name())
    return lemma2syns


def to_wn_pos(ptb_tag: str) -> Optional[str]:
    
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
    
    if not text:
        return ""
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_to_words(text: str, tokenizer_nlp) -> List[str]:
   
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



# Named-entity span alignment (DBpedia char offsets -> word indices)

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



# Medoid (WSI cluster) loading

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



# SQLite index 

def init_db(db_path: str, store_doc_terms: bool = False):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    if store_doc_terms:
        cur.executescript("""
            CREATE TABLE doc_terms ( pid TEXT PRIMARY KEY, blob BLOB NOT NULL ) WITHOUT ROWID;
            CREATE TABLE doc_text  ( pid TEXT PRIMARY KEY, text TEXT ) WITHOUT ROWID;
        """)
    conn.commit()
    return conn
