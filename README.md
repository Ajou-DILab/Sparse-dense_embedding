#  SemSpEm: Semantic Sparase Embedding for Document Retrieval

This is the repository for our paper **SemSpEm: Semantic Sparse Embedding for Document Retrieval**.

---

## Environment

```bash
git clone https://github.com/Ajou-DILab/Sparse-dense_embedding.git
cd Sparse-dense_embedding
```

Tested with Python 3.10+. Install the Python dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

NLTK corpora (`wordnet`, `punkt`, `omw-1.4`, `stopwords`,
`averaged_perceptron_tagger`) are downloaded automatically on first run.

### DBpedia Spotlight (required for the NE stage)

Both indexing and evaluation call a **DBpedia Spotlight REST endpoint** for
named-entity recognition. The endpoint **must be identical** at index time and
query time, otherwise NE terms will not line up. The default is
`http://localhost:2222/rest`.

The simplest way to run one locally is the official Docker image:

```bash
docker run -tid --restart unless-stopped --name dbpedia-spotlight.en \
    -p 2222:80 dbpedia/dbpedia-spotlight spotlight.sh en
```

(For environments where Docker is unavailable, e.g. Colab, the Spotlight JAR
or the public `api.dbpedia-spotlight.org` endpoint can be used instead — adjust
`--dbpedia_endpoint` accordingly.)

---

## Datasets

### WSD Training Dataset
WordNet groups nouns, verbs, adjectives and adverbs into sets of cognitive
synonyms (synsets). **SemCor 3.0**, a large-scale sense-annotated corpus
(226,036 annotated word instances), is used to train the context encoder. See
<https://wordnet.princeton.edu/> for WordNet, and the SemCor distribution for
the training corpus.

Training (`Model/Train_Bi_WSD.py`) expects a CSV of SemCor samples with columns
`context_tokens`, `target_span`, `synset_id` (see `Dataset/dataset.py`).

WSD benchmarks (SE2, SE3, SE07, SE13, SE15) from the Senseval/SemEval
competitions are used to evaluate WSD quality against baselines.

### Retrieval Dataset

We use the **MS MARCO passage ranking** collection, in the standard TSV format:

Download from the official MS MARCO site: <https://microsoft.github.io/msmarco/>.

## Test 

### Model Download

Place the following where you like and point the scripts at them with the
corresponding flags:

| File | Flag | Description |
| --- | --- | --- |
| `best_bi_encoder.pt` | `--ctx_ckpt` | trained WSD bi-encoder checkpoint |
| `wordnet_gloss_embeddings.pt` | `--gloss_vec` | precomputed WordNet gloss embeddings |
| `medoids.pkl` | `--medoids` | WSI medoid cluster centers |

**Model download:** the trained checkpoint and precomputed embeddings areavailable from Google Drive —

[Model and gloss embeddings Download](https://drive.google.com/drive/folders/1kZYOX9WtIq6lnWkiHdMNIapknyiUDiKr?usp=drive_link)

---


### 1. Get the partial subset (for fast end-to-end runs)

To validate the whole pipeline (index + search) in under an hour, we provide a small **partial
subset** (~50k passages) that keeps the passages small sample dev queries.

**Partial subset download:** available from Google Drive —

You have to download msmarco_partial folder for run_partial_pipeline.py emplementation.

[Partial subset download](https://drive.google.com/drive/folders/1MPkf9djEDEg1FBpn_4BLZJxJxKebd69y?usp=drive_link)

---

### 2. Run the partial subset pipeline

#### Partial subset, index + search in one command

For the ~50k-passage partial subset, `run_partial_pipeline.py` indexes the
collection and then runs query mapping, search, and evaluation in a single
process (it imports the indexer and the evaluation helpers internally, so the
mapping is identical on both sides). The whole run finishes in well under an
hour on a single GPU.

```bash
python Evaluation/run_partial_pipeline.py \
    --partial_dir "PARTIAL_SUBSET_PATH" \
    --db_path     "DB_PATH" \
    --ctx_ckpt    "MODEL_PATH" \
    --gloss_vec   "GLOSS_EMBEDDINGS_PATH" \
    --medoids     "MEDOIDS_PATH" \
    --dbpedia_endpoint http://localhost:2222/rest \
    --sample_n    3
```

(The index is always built with `doc_terms` stored, so the `--sample_n`
per-query inspection works out of the box. Add `--skip_indexing` to re-run
search against an index you already built.)

## Citation
