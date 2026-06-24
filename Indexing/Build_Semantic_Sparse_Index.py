"""
Build a sparse (sense/entity-aware) inverted index over an MS MARCO-style
passage collection.

Pipeline per passage:
  1. Named entity recognition (DBpedia Spotlight) -> NE::{type}::{entity} terms.
  2. Word Sense Disambiguation against WordNet senses (via a trained
     SpanContextEncoder + precomputed gloss embeddings) for tokens not
     covered by an NE span.
  3. Word Sense Induction (medoid clustering) as a fallback for tokens that
     WSD could not confidently map to a WordNet sense.

The resulting per-passage term frequencies are written to a SQLite database
(postings list + doc-length table), which downstream sparse retrieval (e.g.
an SBM25-style scorer) reads directly.
"""

import argparse
import os
import sqlite3
import sys
import zlib
from collections import Counter, defaultdict
from contextlib import closing
from functools import partial

import nltk
import orjson
import spacy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Make the repo root and the Dataset/ and Model/ package dirs importable
# regardless of the working directory, so `python Indexing/Build_...py` works
# after a fresh `git clone` on any platform.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT,
           os.path.join(_REPO_ROOT, "Dataset"),
           os.path.join(_REPO_ROOT, "Model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import MSMarcoTSVDataset, passage_collate_fn
from model import SpanContextEncoder
from utils import (build_char_to_word_idx, build_lemma_to_synsets,
                    ensure_nltk, get_ne_word_indices, init_db,
                    lemmatize_tokens, load_medoids, to_wn_pos)

nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def parse_args():
    p = argparse.ArgumentParser(description="Build the SEMSPEM sparse index")
    p.add_argument("--input_tsv", default="./data/ms_marco/collection.tsv",
                    help="MS MARCO-style collection.tsv to index")
    p.add_argument("--ctx_ckpt_path", default="./checkpoints/best_bi_encoder.pt",
                    help="Trained SpanContextEncoder checkpoint")
    p.add_argument("--gloss_vec_path", default="./data/wordnet_gloss_embeddings.pt",
                    help="Precomputed WordNet gloss embeddings")
    p.add_argument("--medoids_path", default="./data/medoids.pkl",
                    help="WSI medoid cluster file")
    p.add_argument("--output_db", default="./output/msmarco_passage_index.sqlite",
                    help="Output SQLite index path")
    p.add_argument("--dbpedia_endpoint", default="http://localhost:2222/rest",
                    help="DBpedia Spotlight REST endpoint. Must match the "
                         "endpoint passed to evaluate_retrieval.py, or NE "
                         "terms will not line up between indexing and query time.")
    p.add_argument("--store_doc_terms", action="store_true",
                    help="Also store per-passage mapped term->tf dicts and raw "
                         "text (doc_terms / doc_text tables). Needed for the "
                         "qualitative per-query inspection in evaluate_retrieval.py "
                         "(--sample_n). Slightly increases index size.")
    return p.parse_args()


# ──────────────────── Hyperparameters (args-independent) ────────────────────
PRETRAINED_MODEL = "bert-base-uncased"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG_SAMPLE_N = 10
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

POSTINGS_BATCH_SIZE = 50_000

if os.name == "nt":
    NUM_WORKERS = 0
    PERSISTENT_WORKERS = False

# ──────────────────── Global State ────────────────────
# Populated by run_indexing(args) at run time. Kept at module scope (rather than
# passed around) so the existing mapping code below, which references them as
# globals, keeps working unchanged.
fast_tokenizer_nlp = spacy.blank("en")
tokenizer = None
STOP = set()
medoids_tensor = None           # single POS-agnostic medoid tensor (WSI)
lemma2syns = defaultdict(list)   # lemma -> [synset names]; built in run_indexing
synset2emb = {}                  # synset name -> gloss embedding; loaded in run_indexing


# ──────────────────── Main Indexing Pipeline ────────────────────
def run_indexing(args):
    """Build the semantic sparse index described by `args`.

    `args` is an argparse.Namespace with at least:
        input_tsv, ctx_ckpt_path, gloss_vec_path, medoids_path, output_db,
        dbpedia_endpoint, store_doc_terms
    Importable: no work happens at module-import time, so other scripts (e.g.
    the partial-subset pipeline) can `from Build_Semantic_Sparse_Index import
    run_indexing` and call it directly.
    """
    global STOP, tokenizer, medoids_tensor, lemma2syns, synset2emb

    INPUT_TSV_PATH = args.input_tsv
    CTX_CKPT_PATH  = args.ctx_ckpt_path
    GLOSS_VEC_PATH = args.gloss_vec_path
    MEDOIDS_PATH   = args.medoids_path
    OUT_SQLITE_DB  = args.output_db

    ensure_nltk()
    if not lemma2syns:
        lemma2syns = build_lemma_to_synsets()

    # gloss embeddings (loaded here, not at import time)
    if os.path.exists(GLOSS_VEC_PATH):
        synset2emb = torch.load(GLOSS_VEC_PATH, map_location=DEVICE)
    else:
        synset2emb = {}

    medoids_tensor = load_medoids(MEDOIDS_PATH, DEVICE)

    from nltk.corpus import stopwords
    STOP = set(stopwords.words('english')) | {"of", "the", "and", "in", "on", "for", "to", "a", "an"}

    print("Loading spaCy with DBpedia Spotlight...")
    import spacy_dbpedia_spotlight  # noqa: F401  (registers the spaCy pipeline component)
    nlp_spotlight = spacy.load("en_core_web_sm", disable=["textcat", "lemmatizer", "ner"])
    nlp_spotlight.add_pipe(
        'dbpedia_spotlight',
        config={
            'dbpedia_rest_endpoint': args.dbpedia_endpoint,
            'confidence': 0.6,
            'support': 20
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, use_fast=True)

    ds = MSMarcoTSVDataset(INPUT_TSV_PATH, sample_limit=None)
    collate = partial(passage_collate_fn, tokenizer_nlp=fast_tokenizer_nlp)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                    persistent_workers=PERSISTENT_WORKERS)

    ctx = SpanContextEncoder(PRETRAINED_MODEL, device=DEVICE)
    if os.path.exists(CTX_CKPT_PATH):
        full_state = torch.load(CTX_CKPT_PATH, map_location=DEVICE)
        ctx_only = {k.replace("ctx_enc.encoder.", "").replace("encoder.", ""): v
                    for k, v in full_state.items() if "encoder." in k}
        ctx.encoder.load_state_dict(ctx_only, strict=False)
    ctx.encoder.eval()

    os.makedirs(os.path.dirname(OUT_SQLITE_DB) or ".", exist_ok=True)

    with closing(init_db(OUT_SQLITE_DB, store_doc_terms=args.store_doc_terms)) as conn:
        cur = conn.cursor()
        sample_out = []
        total_docs = 0

        print(f"Starting Indexing on Device: {DEVICE}")
        cur.execute("BEGIN;")

        with torch.no_grad():
            for pids, words_batch, clean_texts_batch, raw_texts_batch in tqdm(dl, desc="Mapping passages"):

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=(DEVICE.type == "cuda")):
                    enc = tokenizer(words_batch, is_split_into_words=True,
                                    return_tensors="pt", padding=True,
                                    truncation=True).to(DEVICE, non_blocking=True)
                    last_hidden = ctx.encoder(**enc).last_hidden_state

                # Run DBpedia Spotlight NE recognition
                docs = []
                for text in raw_texts_batch:
                    if not text.strip():
                        docs.append(nlp_spotlight.make_doc(text))
                        continue
                    try:
                        doc = nlp_spotlight(text)
                        docs.append(doc)
                    except ValueError:
                        docs.append(nlp_spotlight.make_doc(text))
                    except Exception:
                        docs.append(nlp_spotlight.make_doc(text))

                rows_to_insert = []
                conf_rows_to_insert = []
                doc_terms_rows = []   # (pid, zlib(orjson(term->tf)))  -- only if store_doc_terms
                doc_text_rows = []    # (pid, raw_text)                -- only if store_doc_terms

                for bi, pid in enumerate(pids):
                    total_docs += 1
                    words_i = words_batch[bi]
                    clean_text_i = clean_texts_batch[bi]
                    word_ids = enc.word_ids(batch_index=bi)
                    hidden_i = last_hidden[bi]
                    doc = docs[bi]

                    # Map NE character offsets to word indices.
                    #
                    # DBpedia Spotlight reports entity spans as character
                    # offsets into the raw text, while `words_i` is tokenized
                    # from the cleaned text. We build a char-offset ->
                    # word-index map so that only the tokens actually covered
                    # by an entity span are skipped during WSD/WSI.
                    char_to_widx = build_char_to_word_idx(words_i, clean_text_i)
                    ne_word_idx_to_skip = set()
                    ne_terms = []
                    # NE confidence: accumulate DBpedia similarityScore values
                    ne_sim_scores = []

                    for ent in doc.ents:
                        ent_text = ent.text
                        if not ent_text:
                            continue

                        types_raw = (ent._.dbpedia_raw_result.get('@types', '')
                                     if hasattr(ent._, 'dbpedia_raw_result') else '')

                        parsed_type = "ETC"
                        if types_raw:
                            t_list = [t.split(':')[-1] for t in types_raw.split(',')
                                      if 'DBpedia:' in t]
                            if t_list:
                                parsed_type = t_list[0]

                        # Convert char offsets to word indices and add to skip set
                        ent_word_indices = get_ne_word_indices(
                            ent.start_char, ent.end_char, char_to_widx
                        )
                        ne_word_idx_to_skip.update(ent_word_indices)

                        formatted_ent = ent_text.replace(' ', '_')

                        # similarityScore: DBpedia's confidence (0-1) for this entity mapping
                        if hasattr(ent._, 'dbpedia_raw_result'):
                            raw = ent._.dbpedia_raw_result
                            score = raw.get('@similarityScore', None)
                            if score is not None:
                                try:
                                    ne_sim_scores.append(float(score))
                                except (ValueError, TypeError):
                                    pass

                        # Untyped entities still get indexed, under the NE::ETC:: bucket
                        if parsed_type == "ETC" or not parsed_type:
                            ne_terms.append(f"NE::ETC::{formatted_ent}")
                        else:
                            ne_terms.append(f"NE::{parsed_type}::{formatted_ent}")

                    # Two-stage WSD -> WSI pipeline.
                    #
                    # Stage 1 (WSD): map to a WordNet synset via cosine
                    # similarity against precomputed gloss embeddings.
                    # Stage 2 (WSI): for tokens where WSD found no candidate
                    # synset, fall back to the nearest medoid cluster.
                    #
                    # Tokens with no WordNet POS (function words, etc.) are
                    # skipped entirely.
                    lemmas, tags = lemmatize_tokens(words_i)
                    wsd_terms = []
                    # WSD/WSI confidence: best cosine similarity at mapping time
                    wsd_wsi_sim_scores = []
                    seen = set()
                    unique_word_indices = [w for w in word_ids if w is not None]

                    for w in dict.fromkeys(unique_word_indices):
                        if w in seen:
                            continue
                        seen.add(w)

                        lemma = lemmas[w].lower()
                        if lemma in STOP or len(lemma) < 2:
                            continue

                        if w in ne_word_idx_to_skip:
                            continue

                        wn_p = to_wn_pos(tags[w])
                        if not wn_p:
                            continue

                        span_mask = torch.tensor(
                            [idx == w for idx in word_ids], device=DEVICE
                        )
                        if span_mask.sum().item() == 0:
                            # BERT token alignment failure; shouldn't happen in practice
                            continue

                        emb = hidden_i[span_mask].mean(0)

                        # Stage 1: WSD via WordNet synset gloss embeddings
                        cands = lemma2syns.get(lemma, [])
                        cands = [s for s in cands if s in synset2emb]
                        if cands:
                            cand_embs = torch.stack([synset2emb[s] for s in cands])
                            sims = F.cosine_similarity(emb.unsqueeze(0), cand_embs)
                            best_idx_wsd = sims.argmax().item()
                            best_sim = sims[best_idx_wsd].item()
                            wsd_terms.append(cands[best_idx_wsd])
                            wsd_wsi_sim_scores.append(best_sim)
                            continue  # WSD succeeded; skip WSI fallback

                        # Stage 2: WSI via the global medoid clusters
                        if medoids_tensor is not None:
                            sims = F.cosine_similarity(emb.unsqueeze(0), medoids_tensor)
                            best_idx = sims.argmax().item()
                            best_sim = sims[best_idx].item()
                            wsd_terms.append(f"WSI::CLU::{best_idx}")
                            wsd_wsi_sim_scores.append(best_sim)
                        else:
                            print(f"[WARN] medoids_tensor not loaded "
                                  f"(lemma='{lemma}', pid={pid})")

                    terms_all = ne_terms + wsd_terms
                    tf_counter = Counter(terms_all)
                    rows_to_insert.extend(
                        [(term, pid, int(tf)) for term, tf in tf_counter.items()]
                    )

                    # Optional: keep per-passage mapped terms + raw text for
                    # qualitative inspection at query time.
                    if args.store_doc_terms:
                        doc_terms_rows.append(
                            (pid, sqlite3.Binary(zlib.compress(orjson.dumps(dict(tf_counter)))))
                        )
                        doc_text_rows.append((pid, raw_texts_batch[bi]))

                    # Aggregate per-passage mapping confidence
                    ne_avg = (sum(ne_sim_scores) / len(ne_sim_scores)
                              if ne_sim_scores else None)
                    wsd_wsi_avg = (sum(wsd_wsi_sim_scores) / len(wsd_wsi_sim_scores)
                                   if wsd_wsi_sim_scores else None)
                    conf_rows_to_insert.append((
                        pid,
                        ne_avg,          len(ne_sim_scores),
                        wsd_wsi_avg,     len(wsd_wsi_sim_scores),
                    ))

                    if len(sample_out) < DEBUG_SAMPLE_N:
                        sample_out.append({"pid": pid, "terms": sorted(tf_counter.keys())})

                cur.executemany(
                    "INSERT INTO tmp_tf(term, pid, tf) VALUES (?, ?, ?)",
                    rows_to_insert
                )
                cur.executemany(
                    """INSERT INTO passage_confidence
                       (pid, ne_conf_avg, ne_conf_cnt, wsd_wsi_conf_avg, wsd_wsi_conf_cnt)
                       VALUES (?, ?, ?, ?, ?)""",
                    conf_rows_to_insert
                )

                if args.store_doc_terms:
                    cur.executemany("INSERT INTO doc_terms(pid, blob) VALUES (?, ?)", doc_terms_rows)
                    cur.executemany("INSERT INTO doc_text(pid, text) VALUES (?, ?)", doc_text_rows)

                if total_docs % (BATCH_SIZE * 1000) == 0:
                    conn.commit()
                    cur.execute("BEGIN;")

        cur.execute("COMMIT;")

        print("Calculating DocLen and Meta...")
        cur.executescript("""
            DELETE FROM doclen;
            INSERT INTO doclen(pid, dl) SELECT pid, SUM(tf) AS dl FROM tmp_tf GROUP BY pid;
        """)
        conn.commit()

        N = cur.execute("SELECT COUNT(*) FROM doclen").fetchone()[0]
        avgdl_row = cur.execute("SELECT AVG(dl) FROM doclen").fetchone()
        avgdl = float(avgdl_row[0] if avgdl_row and avgdl_row[0] is not None else 0.0)

        cur.execute("DELETE FROM meta;")
        cur.executemany("INSERT INTO meta(k, v) VALUES (?, ?)",
                        [("N", str(N)), ("avgdl", f"{avgdl:.6f}")])
        conn.commit()

        print("Building Postings Table...")
        cur.execute("DELETE FROM postings;")
        conn.commit()

        cur.execute("BEGIN;")
        fetch_cur = conn.cursor()
        fetch_cur.execute(
            "SELECT term, pid, SUM(tf) FROM tmp_tf GROUP BY term, pid ORDER BY term"
        )

        batch = []
        current_term = None
        current_plist = []

        for row in tqdm(fetch_cur, desc="Postings Construction"):
            term, pid, tf = row
            if current_term != term:
                if current_term is not None:
                    df = len(current_plist)
                    blob = sqlite3.Binary(zlib.compress(orjson.dumps(current_plist)))
                    batch.append((current_term, df, blob))
                    if len(batch) >= POSTINGS_BATCH_SIZE:
                        cur.executemany(
                            "INSERT INTO postings(term, df, blob) VALUES (?, ?, ?)", batch
                        )
                        batch.clear()
                current_term = term
                current_plist = []
            current_plist.append([str(pid), int(tf)])

        if current_term is not None:
            df = len(current_plist)
            blob = sqlite3.Binary(zlib.compress(orjson.dumps(current_plist)))
            batch.append((current_term, df, blob))

        if batch:
            cur.executemany(
                "INSERT INTO postings(term, df, blob) VALUES (?, ?, ?)", batch
            )
        cur.execute("COMMIT;")

        # ──────────────────────────────────────────────────────────────────
        # Final index statistics
        #
        # "mapping count"  = tmp_tf SUM(tf) -> total occurrences of a bucket
        #                     across the whole collection
        # "unique terms"   = distinct terms in `postings` matching that bucket
        #
        # Buckets:
        #   NE (typed):     NE::{type}::   (excludes ETC)
        #   NE (untyped):   NE::ETC::
        #   WSD:            synset form (e.g. bank.n.01) - no NE/WSI prefix
        #   WSI:            WSI::CLU::{idx}  (single global, POS-agnostic clustering)
        # ──────────────────────────────────────────────────────────────────
        print("\nCalculating Final Index Statistics...")

        cur.execute("""
            SELECT COALESCE(SUM(tf), 0) FROM tmp_tf
            WHERE term LIKE 'NE::%' AND term NOT LIKE 'NE::ETC::%'
        """)
        ne_total_mappings = int(cur.fetchone()[0])
        cur.execute("""
            SELECT COUNT(*) FROM postings
            WHERE term LIKE 'NE::%' AND term NOT LIKE 'NE::ETC::%'
        """)
        ne_unique_terms = cur.fetchone()[0]

        cur.execute("""
            SELECT COALESCE(SUM(tf), 0) FROM tmp_tf
            WHERE term LIKE 'NE::ETC::%'
        """)
        ne_etc_total_mappings = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM postings WHERE term LIKE 'NE::ETC::%'")
        ne_etc_unique_terms = cur.fetchone()[0]

        cur.execute("""
            SELECT COALESCE(SUM(tf), 0) FROM tmp_tf
            WHERE term NOT LIKE 'NE::%' AND term NOT LIKE 'WSI::CLU::%'
        """)
        wsd_total_mappings = int(cur.fetchone()[0])
        cur.execute("""
            SELECT COUNT(*) FROM postings
            WHERE term NOT LIKE 'NE::%' AND term NOT LIKE 'WSI::CLU::%'
        """)
        wsd_unique_terms = cur.fetchone()[0]

        cur.execute("""
            SELECT COALESCE(SUM(tf), 0) FROM tmp_tf
            WHERE term LIKE 'WSI::CLU::%'
        """)
        wsi_total_mappings = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM postings WHERE term LIKE 'WSI::CLU::%'")
        wsi_unique_terms = cur.fetchone()[0]

        total_mappings = ne_total_mappings + ne_etc_total_mappings + wsd_total_mappings + wsi_total_mappings
        total_unique   = ne_unique_terms + ne_etc_unique_terms + wsd_unique_terms + wsi_unique_terms

        print(f"SQLite index saved -> {OUT_SQLITE_DB}")
        print(f"  N={N:,}, avgdl={avgdl:.4f}")
        for s in sample_out:
            print(s)


if __name__ == "__main__":
    run_indexing(parse_args())
