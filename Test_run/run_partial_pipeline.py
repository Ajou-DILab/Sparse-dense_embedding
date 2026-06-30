"""
Partial-subset pipeline: index + search in a single run.

For the small (~50k passage) partial MS MARCO subset, this script runs the
*entire* flow end to end -- build the semantic sparse index, then map queries,
search, and report MRR@10 -- so the whole thing finishes in well
under an hour on a single GPU.

It deliberately contains **no retrieval or indexing logic of its own**. It is
pure glue: it imports

    run_indexing            from  Indexing/Build_Semantic_Sparse_Index.py
    Resources,              from  Evaluation/evaluate.py
    SBM25Searcher,
    load_queries, load_qrels,
    evaluate, print_samples

and wires them together. Because both the indexer and the evaluator share the
same NE -> WSD -> WSI mapping (via utils.py / model.py), query-time mapping is
guaranteed to match index-time mapping.

For the full dev set, build the index once with
Indexing/Build_Semantic_Sparse_Index.py and evaluate with
Evaluation/evaluate.py separately; this combined script is a convenience for
the partial subset only.

------------------------------------------------------------------------------
Example
------------------------------------------------------------------------------
    python Evaluation/run_partial_pipeline.py \
        --partial_dir ./data/ms_marco/msmarco_partial \
        --db_path     ./output/semspem_partial_index.sqlite \
        --ctx_ckpt    ./checkpoints/best_bi_encoder.pt \
        --gloss_vec   ./data/wordnet_gloss_embeddings.pt \
        --medoids     ./data/medoids.pkl \
        --dbpedia_endpoint http://localhost:2222/rest \
        --sample_n    3

Re-run search only (index already built):
    ... add --skip_indexing
"""

import argparse
import os
import sys
from types import SimpleNamespace


# Make the repo root and the sibling package dirs importable regardless of the
# working directory, so this runs after a fresh `git clone` on any platform.

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT,
           os.path.join(_REPO_ROOT, "Dataset"),
           os.path.join(_REPO_ROOT, "Model"),
           os.path.join(_REPO_ROOT, "Indexing"),
           os.path.join(_REPO_ROOT, "Evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Indexing side.
from Build_Semantic_Sparse_Index import run_indexing  # noqa: E402

# Search / evaluation side (all reused; nothing reimplemented here).
import evaluate as ev  # noqa: E402



#  Arguments 

def parse_args():
    p = argparse.ArgumentParser(
        description="Partial-subset SEMSPEM pipeline (index + search in one run)"
    )

    # Partial subset (downloadable; see README)
    p.add_argument("--partial_dir", default="./msmarco_partial",
                   help="Directory holding collection.partial.tsv / "
                        "queries.dev.partial.tsv / qrels.dev.partial.tsv")
    p.add_argument("--collection", default=None,
                   help="Passage TSV (default <partial_dir>/collection.partial.tsv)")
    p.add_argument("--queries", default=None,
                   help="Query TSV (default <partial_dir>/queries.dev.partial.tsv)")
    p.add_argument("--qrels", default=None,
                   help="qrels TSV (default <partial_dir>/qrels.dev.partial.tsv)")

    # Index + model resources
    p.add_argument("--db_path", default="./partial_db.sqlite",
                   help="SQLite index path (written by indexing, read by search)")
    p.add_argument("--ctx_ckpt", default="/best_bi_encoder_wsd.pt")
    p.add_argument("--gloss_vec", default="./wordnet_gloss_embeddings.pt")
    p.add_argument("--medoids", default="./medoids.pkl")
    p.add_argument("--pretrained_model", default="bert-base-uncased")

    
    p.add_argument("--dbpedia_endpoint", default="http://localhost:2222/rest")
    p.add_argument("--dbpedia_confidence", type=float, default=0.6)
    p.add_argument("--dbpedia_support", type=int, default=20)

    
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--top_k", type=int, default=1000)
    p.add_argument("--sample_n", type=int, default=3)
    p.add_argument("--max_queries", type=int, default=None)

    
    p.add_argument("--skip_indexing", action="store_true",
                   help="Skip indexing and search the existing --db_path.")

    args = p.parse_args()

    # Resolve partial-subset file paths.
    if args.collection is None:
        args.collection = os.path.join(args.partial_dir, "collection.partial.tsv")
    if args.queries is None:
        args.queries = os.path.join(args.partial_dir, "queries.dev.partial.tsv")
    if args.qrels is None:
        args.qrels = os.path.join(args.partial_dir, "qrels.dev.partial.tsv")
    return args


def _indexing_args(args):
    """Adapt the unified args into the Namespace that run_indexing() expects.

    run_indexing() reads: input_tsv, ctx_ckpt_path, gloss_vec_path,
    medoids_path, output_db, dbpedia_endpoint, store_doc_terms.
    We always set store_doc_terms=True here so the per-query sample output
    (--sample_n) has the doc_terms / doc_text tables available.
    """
    return SimpleNamespace(
        input_tsv=args.collection,
        ctx_ckpt_path=args.ctx_ckpt,
        gloss_vec_path=args.gloss_vec,
        medoids_path=args.medoids,
        output_db=args.db_path,
        dbpedia_endpoint=args.dbpedia_endpoint,
        store_doc_terms=True,
    )



#  index, then search/evaluate

def main():
    args = parse_args()

    print("=" * 62)
    print("  SEMSPEM Partial Pipeline (index + search)")
    print("=" * 62)
    print(f"  partial_dir : {args.partial_dir}")
    print(f"  collection  : {args.collection}")
    print(f"  queries     : {args.queries}")
    print(f"  qrels       : {args.qrels}")
    print(f"  db_path     : {args.db_path}")

    
    if args.skip_indexing and os.path.exists(args.db_path):
        print(f"\n[1] Indexing skipped; using existing index: {args.db_path}")
    else:
        print("\n" + "=" * 62)
        print("  [1] Passage indexing")
        print("=" * 62)
        run_indexing(_indexing_args(args))

    # [2] Query mapping + SBM25 search + evaluation
    print("\n" + "=" * 62)
    print("  [2] Query mapping + SBM25 search + evaluation")
    print("=" * 62)

    R = ev.Resources(args)
    searcher = ev.SBM25Searcher(args.db_path, k1=args.k1, b=args.b)

    qrels = ev.load_qrels(args.qrels)
    queries = ev.load_queries(args.queries)
    print(f"Loaded {len(queries):,} queries | {len(qrels):,} with judgments")

    results = ev.evaluate(args, R, searcher, queries, qrels)

    #  [3] query qualitative sample
    if args.sample_n > 0:
        ev.print_samples(args, R, searcher, queries, qrels, results["eval_qids"])

    print("\n[done] partial pipeline complete.")


if __name__ == "__main__":
    main()
