#  SemSpEm: Semantic Sparase Embedding for Document Retrieval

This is the repository for our paper **SemSpEm: Semantic Sparse Embedding for Document Retrieval**.

---

## Environment

Every entry-point script adds the repository root (and the `Dataset/` /
`Model/` directories) to `sys.path` at startup, so the scripts can be run from
any working directory after a fresh `git clone`:

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

| File | Format |
| --- | --- |
| `collection.tsv` | `pid \t passage_text` |
| `queries.dev.small.tsv` | `qid \t query_text` |
| `qrels.dev.small.tsv` | `qid \t 0 \t pid \t 1` |

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

> **Model download:** the trained checkpoint and precomputed embeddings are
> available from Google Drive — see the link in the repository's release notes.
> (Replace this line with the actual Drive link when publishing.)

### Retrieval Dataset

## Citation
