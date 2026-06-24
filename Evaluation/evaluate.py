"""
End-to-end retrieval evaluation for the SEMSPEM sparse index.

This script maps each dev query through the *same* NE -> WSD -> WSI pipeline
used at indexing time (see Indexing/Build_Semantic_Sparse_Index.py), scores the
sparse index with an SBM25 scorer, and reports MRR@10 / Recall@k.

It works on any collection/queries/qrels triple in the standard MS MARCO TSV
format -- the full dev set, or a smaller subset. For a fast, self-contained
index + search run over a ~50k-passage subset, see
Evaluation/run_partial_pipeline.py, which imports both the indexer and the
evaluation helpers defined here.

It reuses the shared modules from the repository root:
    utils.py          (text processing, NE span alignment, medoid loading)
    Model/model.py    (SpanContextEncoder)

so that query-time mapping cannot drift from index-time mapping. The core
search/eval pieces (Resources, SBM25Searcher, map_query, evaluate,
print_samples) are importable by other scripts.

------------------------------------------------------------------------------
Qualitative per-query output (--sample_n N)
------------------------------------------------------------------------------
For the first N evaluated queries, prints:
  * the query text and its mapped senses (NE / WSD synset / WSI cluster)
  * the synset + gloss behind each mapped term
  * the top-3 retrieved documents, each with its SBM25 score, its own indexed
    terms, and which of those terms overlap with the query
This requires the index to have been built with --store_doc_terms.

------------------------------------------------------------------------------
Example
------------------------------------------------------------------------------
    python Evaluation/evaluate.py \
        --queries   ./data/ms_marco/queries.dev.small.tsv \
        --qrels     ./data/ms_marco/qrels.dev.small.tsv \
        --db_path   ./output/semspem_index.sqlite \
        --ctx_ckpt  ./checkpoints/best_bi_encoder.pt \
        --gloss_vec ./data/wordnet_gloss_embeddings.pt \
        --medoids   ./data/medoids.pkl \
        --sample_n  3
"""

import argparse
import csv
import math
import os
import sqlite3
import sys
import zlib
from collections import Counter, defaultdict
from typing import List, Tuple

import orjson
import spacy
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from transformers import AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────
# Make the repository root (and the Model/ package dir) importable regardless
# of the current working directory, so `python Evaluation/evaluate.py`
# works after a fresh `git clone` on any platform.
# ──────────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "Model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (  # noqa: E402  (import after sys.path tweak)
    build_char_to_word_idx,
    build_lemma_to_synsets,
    clean_text_for_indexing,
    ensure_nltk,
    get_ne_word_indices,
    lemmatize_tokens,
    load_medoids,
    to_wn_pos,
)
from model import SpanContextEncoder  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
#  Arguments
# ════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SEMSPEM sparse retrieval")

    # Query / qrels inputs. Provide them directly, or point --partial_dir at a
    # directory that contains queries.dev.partial.tsv / qrels.dev.partial.tsv
    # (the layout of the downloadable partial subset) and they will be filled in.
    p.add_argument("--queries", default=None,
                   help="Query TSV path (qid \\t text). Required unless --partial_dir is given.")
    p.add_argument("--qrels", default=None,
                   help="qrels TSV path (qid 0 pid 1). Required unless --partial_dir is given.")
    p.add_argument("--partial_dir", default=None,
                   help="Convenience: a directory holding queries.dev.partial.tsv and "
                        "qrels.dev.partial.tsv; used to fill --queries/--qrels if unset.")

    # Index + model resources.
    p.add_argument("--db_path", default="./output/semspem_index.sqlite",
                   help="SQLite index built by Build_Semantic_Sparse_Index.py")
    p.add_argument("--ctx_ckpt", default="./checkpoints/best_bi_encoder.pt",
                   help="Trained SpanContextEncoder checkpoint")
    p.add_argument("--gloss_vec", default="./data/wordnet_gloss_embeddings.pt",
                   help="Precomputed WordNet gloss embeddings")
    p.add_argument("--medoids", default="./data/medoids.pkl",
                   help="WSI medoid cluster file")
    p.add_argument("--pretrained_model", default="bert-base-uncased")

    # DBpedia Spotlight (must match indexing-time settings).
    p.add_argument("--dbpedia_endpoint", default="http://localhost:2222/rest",
                   help="Must match the endpoint used when building the index.")
    p.add_argument("--dbpedia_confidence", type=float, default=0.6)
    p.add_argument("--dbpedia_support", type=int, default=20)

    # SBM25 params.
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)

    # Evaluation / output knobs.
    p.add_argument("--top_k", type=int, default=1000,
                   help="Depth for Recall@k (MRR is always @10).")
    p.add_argument("--sample_n", type=int, default=3,
                   help="Print detailed mapping/retrieval for the first N queries "
                        "(needs an index built with --store_doc_terms).")
    p.add_argument("--max_queries", type=int, default=None,
                   help="Optional cap on number of evaluated queries (debug).")

    args = p.parse_args()
    _resolve_query_paths(args)
    return args


def _resolve_query_paths(args):
    """Fill args.queries / args.qrels from args.partial_dir when not given.

    Shared by the CLI and by run_partial_pipeline.py so the path-resolution
    logic lives in one place.
    """
    if getattr(args, "partial_dir", None):
        if not args.queries:
            args.queries = os.path.join(args.partial_dir, "queries.dev.partial.tsv")
        if not args.qrels:
            args.qrels = os.path.join(args.partial_dir, "qrels.dev.partial.tsv")
    if not args.queries or not args.qrels:
        raise SystemExit(
            "Both --queries and --qrels are required (or pass --partial_dir "
            "containing queries.dev.partial.tsv / qrels.dev.partial.tsv)."
        )
    return args


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
#  Resource container — loaded once, shared by query mapping
# ════════════════════════════════════════════════════════════════════════════
class Resources:
    def __init__(self, args):
        ensure_nltk()

        from nltk.corpus import stopwords
        self.STOP = set(stopwords.words('english')) | {
            "of", "the", "and", "in", "on", "for", "to", "a", "an"
        }

        self.fast_tokenizer_nlp = spacy.blank("en")

        print("Loading spaCy with DBpedia Spotlight...")
        import spacy_dbpedia_spotlight  # noqa: F401
        self.nlp_spotlight = spacy.load(
            "en_core_web_sm", disable=["textcat", "lemmatizer", "ner"]
        )
        self.nlp_spotlight.add_pipe(
            'dbpedia_spotlight',
            config={
                'dbpedia_rest_endpoint': args.dbpedia_endpoint,
                'confidence': args.dbpedia_confidence,
                'support': args.dbpedia_support,
            }
        )

        print("Loading Context Encoder...")
        self.ctx = SpanContextEncoder(args.pretrained_model, device=DEVICE)
        if os.path.exists(args.ctx_ckpt):
            full_state = torch.load(args.ctx_ckpt, map_location=DEVICE)
            ctx_only = {k.replace("ctx_enc.encoder.", "").replace("encoder.", ""): v
                        for k, v in full_state.items() if "encoder." in k}
            self.ctx.encoder.load_state_dict(ctx_only, strict=False)
            print(f"  Loaded checkpoint: {args.ctx_ckpt}")
        else:
            print(f"  [WARN] checkpoint not found, using base weights: {args.ctx_ckpt}")
        self.ctx.encoder.eval()
        # SpanContextEncoder ships its own tokenizer; keep a direct handle.
        self.tokenizer = self.ctx.tokenizer

        print("Loading WordNet gloss embeddings...")
        self.synset2emb = (torch.load(args.gloss_vec, map_location=DEVICE)
                           if os.path.exists(args.gloss_vec) else {})
        self.lemma2syns = build_lemma_to_synsets()
        print(f"  synset2emb: {len(self.synset2emb):,} | lemma2syns: {len(self.lemma2syns):,}")

        self.medoids_tensor = load_medoids(args.medoids, DEVICE)


# ════════════════════════════════════════════════════════════════════════════
#  Query mapping — identical NE -> WSD -> WSI pipeline as indexing
#  Returns (terms, detail). `detail` is only populated when collect_detail=True
#  (used for the qualitative sample output).
# ════════════════════════════════════════════════════════════════════════════
def map_query(query_text: str, R: Resources, collect_detail: bool = False
              ) -> Tuple[List[str], list]:
    clean_text = clean_text_for_indexing(query_text)
    words = [t.text for t in R.fast_tokenizer_nlp.tokenizer(clean_text)] if clean_text else []
    if not words:
        return [], []

    with torch.no_grad():
        with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                            dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
            enc = R.tokenizer(words, is_split_into_words=True,
                              return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            last_hidden = R.ctx.encoder(**enc).last_hidden_state[0]
            word_ids = enc.word_ids(batch_index=0)

    # ── NE (DBpedia Spotlight) ──────────────────────────────────────────────
    try:
        doc = R.nlp_spotlight(query_text)
    except Exception:
        doc = R.nlp_spotlight.make_doc(query_text)

    char_to_widx = build_char_to_word_idx(words, clean_text)
    ne_word_idx_to_skip = set()
    ne_terms = []
    detail = []

    for ent in doc.ents:
        if not ent.text:
            continue
        types_raw = (ent._.dbpedia_raw_result.get('@types', '')
                     if hasattr(ent._, 'dbpedia_raw_result') else '')
        parsed_type = "ETC"
        if types_raw:
            t_list = [t.split(':')[-1] for t in types_raw.split(',') if 'DBpedia:' in t]
            if t_list:
                parsed_type = t_list[0]

        ent_word_indices = get_ne_word_indices(ent.start_char, ent.end_char, char_to_widx)
        ne_word_idx_to_skip.update(ent_word_indices)

        formatted_ent = ent.text.replace(' ', '_')

        sim_score = None
        if hasattr(ent._, 'dbpedia_raw_result'):
            sc = ent._.dbpedia_raw_result.get('@similarityScore', None)
            if sc is not None:
                try:
                    sim_score = float(sc)
                except (ValueError, TypeError):
                    sim_score = None

        if parsed_type == "ETC" or not parsed_type:
            term = f"NE::ETC::{formatted_ent}"
        else:
            term = f"NE::{parsed_type}::{formatted_ent}"
        ne_terms.append(term)
        if collect_detail:
            detail.append({"term": term, "surface": ent.text, "kind": "NE", "sim": sim_score})

    # ── WSD -> WSI ──────────────────────────────────────────────────────────
    lemmas, tags = lemmatize_tokens(words)
    wsd_terms = []
    seen = set()
    unique_word_indices = [w for w in word_ids if w is not None]

    for w in dict.fromkeys(unique_word_indices):
        if w in seen:
            continue
        seen.add(w)

        lemma = lemmas[w].lower()
        if lemma in R.STOP or len(lemma) < 2:
            continue
        if w in ne_word_idx_to_skip:
            continue
        wn_p = to_wn_pos(tags[w])
        if not wn_p:
            continue

        span_mask = torch.tensor([idx == w for idx in word_ids], device=DEVICE)
        if span_mask.sum().item() == 0:
            continue
        emb = last_hidden[span_mask].mean(0)

        # Stage 1: WSD
        cands = [s for s in R.lemma2syns.get(lemma, []) if s in R.synset2emb]
        if cands:
            cand_embs = torch.stack([R.synset2emb[s] for s in cands])
            sims = F.cosine_similarity(emb.unsqueeze(0), cand_embs)
            best = sims.argmax().item()
            term = cands[best]
            wsd_terms.append(term)
            if collect_detail:
                detail.append({"term": term, "surface": lemma,
                               "kind": "WSD", "sim": float(sims[best].item())})
            continue

        # Stage 2: WSI
        if R.medoids_tensor is not None:
            sims = F.cosine_similarity(emb.unsqueeze(0), R.medoids_tensor)
            best = sims.argmax().item()
            term = f"WSI::CLU::{best}"
            wsd_terms.append(term)
            if collect_detail:
                detail.append({"term": term, "surface": lemma,
                               "kind": "WSI", "sim": float(sims[best].item())})

    return ne_terms + wsd_terms, detail


# ════════════════════════════════════════════════════════════════════════════
#  SBM25 searcher
# ════════════════════════════════════════════════════════════════════════════
class SBM25Searcher:
    def __init__(self, db_path: str, k1: float = 1.2, b: float = 0.75):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Index not found: {db_path}\n"
                f"Build it first with Indexing/Build_Semantic_Sparse_Index.py"
            )
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self.k1, self.b = k1, b

        print("Loading SQLite metadata and DocLens to memory...")
        self.N = int(self.cur.execute("SELECT v FROM meta WHERE k='N'").fetchone()[0])
        self.avgdl = float(self.cur.execute("SELECT v FROM meta WHERE k='avgdl'").fetchone()[0])
        self.doclen = dict(self.cur.execute("SELECT pid, dl FROM doclen").fetchall())
        print(f"  N={self.N:,}, avgdl={self.avgdl:.4f}, doclen entries={len(self.doclen):,}")

        # Detect optional doc_terms / doc_text tables (for --sample_n output).
        tbls = {r[0] for r in self.cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.has_doc_terms = "doc_terms" in tbls and "doc_text" in tbls

    def search(self, mapped_query_terms: List[str], top_k: int = 1000,
               return_scores: bool = False):
        q_counts = Counter(mapped_query_terms)
        scores = defaultdict(float)
        for term in q_counts:
            row = self.cur.execute(
                "SELECT df, blob FROM postings WHERE term=?", (term,)
            ).fetchone()
            if not row:
                continue
            df, blob = row
            idf = math.log(((self.N - df + 0.5) / (df + 0.5)) + 1.0)
            plist = orjson.loads(zlib.decompress(blob))
            for pid_str, tf in plist:
                dl = self.doclen.get(pid_str, self.avgdl)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                )
                scores[pid_str] += idf * tf_norm
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if return_scores:
            return ranked[:top_k]
        return [pid for pid, _ in ranked[:top_k]]

    def get_doc_terms(self, pid: str) -> dict:
        if not self.has_doc_terms:
            return {}
        row = self.cur.execute("SELECT blob FROM doc_terms WHERE pid=?", (pid,)).fetchone()
        return orjson.loads(zlib.decompress(row[0])) if row else {}

    def get_doc_text(self, pid: str) -> str:
        if not self.has_doc_terms:
            return ""
        row = self.cur.execute("SELECT text FROM doc_text WHERE pid=?", (pid,)).fetchone()
        return row[0] if row else ""


# ════════════════════════════════════════════════════════════════════════════
#  synset / gloss description helper (for qualitative output)
# ════════════════════════════════════════════════════════════════════════════
def describe_term(term: str) -> dict:
    info = {"term": term, "category": None, "synset": None, "gloss": None}
    if term.startswith("NE::"):
        parts = term.split("::", 2)
        info["category"] = "NE"
        info["synset"] = f"{parts[1]} / {parts[2]}" if len(parts) == 3 else term
        info["gloss"] = "(named entity - DBpedia)"
    elif term.startswith("WSI::CLU::"):
        info["category"] = "WSI"
        info["synset"] = term
        info["gloss"] = "(word-sense-induction cluster - no WordNet gloss)"
    else:
        info["category"] = "WSD"
        try:
            syn = wn.synset(term)
            info["synset"] = syn.name()
            info["gloss"] = syn.definition()
        except Exception:
            info["synset"] = term
            info["gloss"] = "(gloss lookup failed)"
    return info


# ════════════════════════════════════════════════════════════════════════════
#  Data loading
# ════════════════════════════════════════════════════════════════════════════
def load_qrels(path: str) -> dict:
    qrels = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) >= 4 and int(row[3]) > 0:
                qrels[row[0]].add(row[2])
    return qrels


def load_queries(path: str) -> dict:
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) >= 2:
                queries[row[0]] = row[1]
    return queries


# ════════════════════════════════════════════════════════════════════════════
#  Evaluation (pure definitions; no score adjustments)
# ════════════════════════════════════════════════════════════════════════════
def evaluate(args, R: Resources, searcher: SBM25Searcher,
             queries: dict, qrels: dict) -> dict:
    eval_qids = [qid for qid in queries if qid in qrels]
    if args.max_queries:
        eval_qids = eval_qids[:args.max_queries]

    mrr_sum = recall_sum = 0.0
    valid = no_result = 0

    for qid in tqdm(eval_qids, desc="Evaluating"):
        mapped, _ = map_query(queries[qid], R)
        top = searcher.search(mapped, top_k=args.top_k)
        rel = qrels[qid]
        if not top:
            no_result += 1

        rr = 0.0
        for rank, pid in enumerate(top[:10], start=1):
            if pid in rel:
                rr = 1.0 / rank
                break

        hits = sum(1 for pid in top if pid in rel)
        recall = hits / len(rel) if rel else 0.0

        mrr_sum += rr
        recall_sum += recall
        valid += 1

    mrr10 = mrr_sum / valid if valid else 0.0
    recall_at_k = recall_sum / valid if valid else 0.0

    print("\n" + "=" * 52)
    print("Retrieval Evaluation Results")
    print("=" * 52)
    print(f"  Valid queries evaluated : {valid:>10,}")
    print(f"  Queries with no results : {no_result:>10,}")
    print("-" * 52)
    print(f"  MRR@10                  : {mrr10:>10.4f}")
    print(f"  Recall@{args.top_k:<17}: {recall_at_k:>10.4f}")
    print("=" * 52)

    return {"mrr@10": mrr10, f"recall@{args.top_k}": recall_at_k,
            "valid_queries": valid, "eval_qids": eval_qids}


# ════════════════════════════════════════════════════════════════════════════
#  Qualitative per-query sample output
# ════════════════════════════════════════════════════════════════════════════
def print_samples(args, R, searcher, queries, qrels, eval_qids):
    sample_qids = eval_qids[:args.sample_n]
    print("\n\n" + "#" * 62)
    print(f"  Per-query inspection (first {len(sample_qids)} queries)")
    print("#" * 62)

    if not searcher.has_doc_terms:
        print("\n[NOTE] The index has no doc_terms/doc_text tables, so per-document")
        print("       indexed terms cannot be shown. Rebuild the index with")
        print("       --store_doc_terms to enable full qualitative output.")

    for si, qid in enumerate(sample_qids, start=1):
        qtext = queries[qid]
        mapped, detail = map_query(qtext, R, collect_detail=True)
        ranked = searcher.search(mapped, top_k=3, return_scores=True)
        rel = qrels.get(qid, set())
        q_term_set = set(mapped)

        print("\n" + "-" * 62)
        print(f"[Sample {si}]  QID = {qid}")
        print("-" * 62)
        print(f"* Query text:\n    {qtext}")

        # (1) mapped senses
        print(f"\n* Query mapping ({len(mapped)} terms):")
        if not detail:
            print("    (no mapped terms)")
        for d in detail:
            sim_str = f"{d['sim']:.4f}" if d['sim'] is not None else "N/A"
            print(f"    - [{d['kind']:3}] '{d['surface']}'  ->  {d['term']}")

        # (2) synset / gloss per mapping
        print("\n* synset / gloss per mapped term:")
        for d in detail:
            info = describe_term(d["term"])
            print(f"    - {d['term']}")
            print(f"        synset: {info['synset']}")
            print(f"        gloss : {info['gloss']}")

        # (3) top-3 docs with score + indexed terms
        print("\n* Retrieved Top-3 documents (SBM25):")
        if not ranked:
            print("    (no results)")
        for rank, (pid, score) in enumerate(ranked, start=1):
            tag = "  <-- relevant" if pid in rel else ""
            dtext = searcher.get_doc_text(pid)
            dterms = searcher.get_doc_terms(pid)
            print(f"\n    -- Rank {rank} | PID={pid} | SBM25={score:.4f}{tag}")
            if dtext:
                snippet = (dtext[:160] + "...") if len(dtext) > 160 else dtext
                print(f"       text  : {snippet}")
            if dterms:
                overlap = {t: tf for t, tf in dterms.items() if t in q_term_set}
                others = {t: tf for t, tf in dterms.items() if t not in q_term_set}
                print(f"       indexed terms ({len(dterms)} total):")
                if overlap:
                    ov = ", ".join(f"{t}(tf={tf})" for t, tf in sorted(overlap.items()))
                    print(f"         matched w/ query: {ov}")
                show = list(sorted(others.items()))[:10]
                if show:
                    ot = ", ".join(f"{t}(tf={tf})" for t, tf in show)
                    more = "" if len(others) <= 10 else f"  (+{len(others) - 10} more)"
                    print(f"         others: {ot}{more}")

    print("\n" + "#" * 62)


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    print("=" * 62)
    print("  SEMSPEM Retrieval Evaluation")
    print("=" * 62)
    print(f"  db_path     : {args.db_path}")
    print(f"  queries     : {args.queries}")
    print(f"  qrels       : {args.qrels}")
    print(f"  device      : {DEVICE}")

    R = Resources(args)
    searcher = SBM25Searcher(args.db_path, k1=args.k1, b=args.b)

    qrels = load_qrels(args.qrels)
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries):,} queries | {len(qrels):,} with judgments")

    results = evaluate(args, R, searcher, queries, qrels)

    if args.sample_n > 0:
        print_samples(args, R, searcher, queries, qrels, results["eval_qids"])

    print("\n[done] evaluation complete.")


if __name__ == "__main__":
    main()
