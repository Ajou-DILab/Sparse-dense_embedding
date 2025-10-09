import os, re, math, sqlite3, zlib, orjson
from collections import defaultdict, Counter
from contextlib import closing
from typing import List, Tuple

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import spacy, nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# ──────────────────── 경로 & 하이퍼파라미터 ────────────────────
MSMARCO_DIR         = "C:/Users/USER/Desktop/SSE/dataset/ms_marco"
PRETRAINED_MODEL    = "bert-base-uncased"

CTX_CKPT_PATH       = "C:/Users/USER/Desktop/SSE/save_model/best_bi_encoder_in_batch_0909.pt"
GLOSS_VEC_PATH      = "C:/Users/USER/Desktop/SSE/dataset/wordnet_gloss_embeddings_0909.pt"

BATCH_SIZE          = 64
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_SQLITE_DB       = "C:/Users/USER/Desktop/SSE/dataset/passage_index_1009.sqlite"
DEBUG_SAMPLE_N      = 3   # 샘플 출력 개수

# 성능 관련(필요시 조절)
NUM_WORKERS         = 4
PIN_MEMORY          = True
PERSISTENT_WORKERS  = True
SPACY_N_PROCESS     = 4       # CPU 코어 수에 맞춰
SPACY_PIPE_BATCH    = 256     # 한 번에 NER 돌릴 문장 수
POSTINGS_BATCH_SIZE = 50_000  # postings 테이블 insert 배치 크기

# --- Windows 안전모드: 워커 이슈 있으면 0으로 시작 ---
if os.name == "nt":
    NUM_WORKERS = 0
    PERSISTENT_WORKERS = False

# ──────────────────── NLTK/모델 로딩을 워커 재실행 방지 형태로 ────────────────────
HERE = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
NLTK_DIR = os.path.join(HERE, ".nltk_data")

def ensure_nltk():
    os.makedirs(NLTK_DIR, exist_ok=True)
    if NLTK_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DIR)
    need = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
    ]
    for res_path, pkg in need:
        try:
            nltk.data.find(res_path)
        except LookupError:
            nltk.download(pkg, download_dir=NLTK_DIR, quiet=True)

# ──────────────────── 전역(런타임에 채움) ────────────────────
STOP = set()
nlp = None
tokenizer = None
LEMM = WordNetLemmatizer()

# ──────────────────── 고유명사(NE) 정규화/중첩 해소 ────────────────────
NE_PRIORITY = ["EVENT","ORG","GPE","PERSON","NORP","WORK_OF_ART","PRODUCT","LOC",
               "LAW","FAC","LANGUAGE","DATE","TIME","CARDINAL","QUANTITY","ORDINAL",
               "PERCENT","MONEY","UNK"]
NE_RANK = {t:i for i,t in enumerate(NE_PRIORITY)}
_ARTICLES = {"the","a","an"}

def canon_ne(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[-/]", " ", s)
    toks = [t for t in re.split(r"\s+", s) if t]
    if toks and toks[0] in _ARTICLES:
        toks = toks[1:]
    toks = [re.sub(r"[^a-z0-9]+","", t) for t in toks]
    s = "_".join([t for t in toks if t])
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unk"

def resolve_overlaps(entities):
    entities = sorted(entities,
                      key=lambda e: ((e["end"]-e["start"]),
                                     -NE_RANK.get(e["label"], 999)),
                      reverse=True)
    kept, taken = [], []
    for e in entities:
        if any(not (e["end"]<=s or e["start"]>=t) for (s,t) in taken):
            continue
        kept.append(e)
        taken.append((e["start"], e["end"]))
    return kept

def extract_ne_from_doc(doc):
    """
    doc(spaCy Doc) → (ne_items, ne_char_spans)
      - ne_items: [{"term": "NE::<TYPE>::<canon>", "span": (start,end)}, ...]
      - ne_char_spans: [(start,end), ...]
    """
    raw = []
    # (a) spaCy NER
    for ent in doc.ents:
        raw.append({"start": ent.start_char, "end": ent.end_char,
                    "text": ent.text, "label": ent.label_})
    # (b) NNP 연속 백오프
    toks = list(doc)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.tag_ in ("NNP","NNPS") and re.match(r'^[A-Z]', t.text):
            j = i
            while j+1 < len(toks) and toks[j+1].tag_ in ("NNP","NNPS"):
                j += 1
            span_toks = [w.text for w in toks[i:j+1] if w.text.lower() not in STOP]
            if span_toks and len(span_toks) > 1:
                start = toks[i].idx
                end   = toks[j].idx + len(toks[j].text)
                raw.append({"start": start, "end": end,
                            "text": " ".join(span_toks), "label": "UNK"})
            i = j + 1
        else:
            i += 1

    resolved = resolve_overlaps(raw)

    best_type = {}
    for e in resolved:
        canon = canon_ne(e["text"])
        if not canon:
            continue
        t_new = e["label"] if e["label"] in NE_RANK else "UNK"
        t_old = best_type.get(canon)
        if t_old is None or NE_RANK[t_new] < NE_RANK[t_old]:
            best_type[canon] = t_new

    ne_items = []
    for e in resolved:
        canon = canon_ne(e["text"])
        if not canon:
            continue
        ne_type = best_type.get(canon, "UNK")
        term = f"NE::{ne_type}::{canon}"
        ne_items.append({"term": term, "span": (e["start"], e["end"])})
    ne_spans = [(e["start"], e["end"]) for e in resolved]
    return ne_items, ne_spans

# ──────────────────── 토큰/태깅 유틸 ────────────────────
def to_wn_pos(ptb_tag: str):
    if not ptb_tag: return None
    t = ptb_tag[0]
    if t == 'N': return wn.NOUN
    if t == 'V': return wn.VERB
    if t == 'J': return wn.ADJ
    if t == 'R': return wn.ADV
    return None

def tokenize_to_words(text:str):
    return [t.text for t in nlp(text)]

def lemmatize_tokens(words: List[str]) -> Tuple[List[str], List[str]]:
    tagged = pos_tag(words)
    L = WordNetLemmatizer()
    lemmas, tags = [], []
    for w, tag in tagged:
        wp = to_wn_pos(tag)
        w_low  = w.lower()
        if wp:
            lemmas.append(L.lemmatize(w_low, pos=wp))
        else:
            lemmas.append(L.lemmatize(w_low))
        tags.append(tag)
    return lemmas, tags

def token_in_any_span(start, end, spans):
    for s,e in spans:
        if not (end <= s or start >= e):
            return True
    return False

# ──────────────────── SpanContextEncoder ────────────────────
class SpanContextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder   = AutoModel.from_pretrained(pretrained_model_name).to(self.device)

# ──────────────────── 데이터셋 ────────────────────
class PassageDataset(Dataset):
    def __init__(self, collection_path, sample_limit=None):
        self.passages={}
        with open(collection_path,encoding="utf-8") as f:
            for line in f:
                pid, txt = line.strip().split("\t",1)
                self.passages[pid]=txt
        self.pids=list(self.passages.keys())
        if sample_limit:
            self.pids=self.pids[:sample_limit]
    def __len__(self): return len(self.pids)
    def __getitem__(self,idx):
        pid=self.pids[idx]
        return pid, self.passages[pid]

def collate_fn(batch):
    pids, texts = zip(*batch)
    words = [tokenize_to_words(t) for t in texts]
    return list(pids), words, list(texts)

# ──────────────────── WordNet 사전 자료 ────────────────────
lemma2syns = defaultdict(list)
for _syn in wn.all_synsets():
    for _l in _syn.lemmas():
        lemma2syns[_l.name().lower()].append(_syn.name())
synset2emb = torch.load(GLOSS_VEC_PATH, map_location="cpu")  # {synset → tensor}

# ──────────────────── SQLite 초기화 ────────────────────
def init_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")      # ~200MB
    cur.execute("PRAGMA mmap_size=268435456;")     # 256MB
    cur.executescript("""
    CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE doclen (pid TEXT PRIMARY KEY, dl INTEGER NOT NULL) WITHOUT ROWID;
    -- 최종 포스팅 테이블 (BM25 인퍼런스 호환: blob = zlib(orjson([[pid, tf], ...])))
    CREATE TABLE postings (
        term TEXT PRIMARY KEY,
        df   INTEGER NOT NULL,
        blob BLOB NOT NULL
    ) WITHOUT ROWID;
    -- 스트리밍 적재용 임시 테이블 (문서 단위 TF 누적; pos 저장 안 함!)
    CREATE TABLE tmp_tf (
        term TEXT NOT NULL,
        pid  TEXT NOT NULL,
        tf   INTEGER NOT NULL
    );
    CREATE INDEX idx_tmp_tf_term ON tmp_tf(term);
    CREATE INDEX idx_tmp_tf_pid  ON tmp_tf(pid);
    """)
    conn.commit()
    return conn

# ──────────────────── 메인 인덱싱 ────────────────────
def main():
    # 1) NLTK 리소스 보장 (워커 import 재다운로드 방지)
    ensure_nltk()

    # 2) STOP 세트 초기화
    from nltk.corpus import stopwords
    global STOP
    STOP = set(stopwords.words('english')) | {"of","the","and","in","on","for","to","a","an"}

    # 3) spaCy / tokenizer 로드 (한 번만)
    global nlp, tokenizer
    nlp = spacy.load("en_core_web_sm", disable=["parser","textcat","lemmatizer"])
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, use_fast=True)

    # 4) DataLoader
    collection_path = os.path.join(MSMARCO_DIR,"collection.tsv")
    ds = PassageDataset(collection_path)
    dl = DataLoader(
        ds, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    # 5) context-encoder 가중치 로드
    ctx = SpanContextEncoder(PRETRAINED_MODEL, device=DEVICE)
    full_state = torch.load(CTX_CKPT_PATH, map_location=DEVICE)
    ctx_only = {}
    for k, v in full_state.items():
        if k.startswith("ctx_enc.encoder."):
            ctx_only[k.replace("ctx_enc.encoder.","")] = v
        elif k.startswith("encoder."):
            ctx_only[k.replace("encoder.","")] = v
    ctx.encoder.load_state_dict(ctx_only, strict=False)
    ctx.encoder.eval()

    # 6) DB
    os.makedirs(os.path.dirname(OUT_SQLITE_DB), exist_ok=True)
    with closing(init_db(OUT_SQLITE_DB)) as conn:
        cur = conn.cursor()

        sample_out=[]
        total_docs = 0

        # ───── 1) 문서 단위로 용어→TF를 만들어 tmp_tf에 적재 (pos/-1 저장 금지) ─────
        cur.execute("BEGIN;")  # 대트랜잭션
        with torch.no_grad():
            for pids, words_batch, texts_batch in tqdm(dl, desc="Mapping passages"):
                # spaCy 병렬 NER
                docs = list(nlp.pipe(texts_batch, batch_size=SPACY_PIPE_BATCH, n_process=SPACY_N_PROCESS))

                # BERT 인퍼런스 (+AMP)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE.type=="cuda")):
                    enc = tokenizer(words_batch, is_split_into_words=True,
                                    return_tensors="pt", padding=True,
                                    truncation=True).to(DEVICE, non_blocking=True)
                    last_hidden = ctx.encoder(**enc).last_hidden_state  # [B,L,H]

                rows_to_insert = []  # [(term, pid, tf), ...]

                for bi, (pid, doc) in enumerate(zip(pids, docs)):
                    total_docs += 1
                    text_i = texts_batch[bi]
                    words_i = words_batch[bi]
                    word_ids = enc.word_ids(batch_index=bi)
                    hidden_i = last_hidden[bi]

                    # 토큰별 문자 스팬
                    word_spans = [(t.idx, t.idx+len(t.text)) for t in doc]
                    span_align_ok = (len(words_i) == len(word_spans))

                    # ── NE 수집: 등장 횟수만 TF에 반영 ──
                    ne_items, ne_spans = extract_ne_from_doc(doc)
                    ne_terms = []
                    if span_align_ok:
                        for item in ne_items:
                            term = item["term"]
                            s, e = item["span"]
                            for w, (ws, we) in enumerate(word_spans):
                                if not (we <= s or ws >= e):
                                    ne_terms.append(term)
                    else:
                        # 정렬 불가 시 최소 1회
                        ne_terms.extend([it["term"] for it in ne_items])

                    # ── WSD: NE 내부 토큰은 스킵 ──
                    lemmas, tags = lemmatize_tokens(words_i)
                    wsd_terms = []
                    seen=set()
                    unique_word_indices = [w for w in word_ids if w is not None]
                    for w in dict.fromkeys(unique_word_indices):
                        if w in seen:
                            continue
                        seen.add(w)

                        if span_align_ok:
                            ws, we = word_spans[w]
                            if token_in_any_span(ws, we, ne_spans):
                                continue

                        span_mask = torch.tensor([idx==w for idx in word_ids], device=DEVICE)
                        if span_mask.sum().item() == 0:
                            continue
                        emb = hidden_i[span_mask].mean(0).cpu()

                        lemma = lemmas[w].lower()
                        cands = lemma2syns.get(lemma, [])
                        if not cands:
                            continue
                        cands = [s for s in cands if s in synset2emb]
                        if not cands:
                            continue

                        cand_embs = torch.stack([synset2emb[s] for s in cands])  # CPU tensor
                        sims = F.cosine_similarity(emb.unsqueeze(0), cand_embs)
                        best = cands[sims.argmax().item()]
                        wsd_terms.append(best)

                    # 이 문서의 term 멀티셋 → TF로 압축 (★ 핵심: pos/-1 저장 금지)
                    terms_all = ne_terms + wsd_terms
                    tf_counter = Counter(terms_all)  # term -> tf

                    rows_to_insert.extend([(term, pid, int(tf)) for term, tf in tf_counter.items()])

                    # 샘플
                    if len(sample_out) < DEBUG_SAMPLE_N:
                        sample_out.append({"pid": pid, "terms": sorted(tf_counter.keys())})

                # 미니배치 삽입
                cur.executemany("INSERT INTO tmp_tf(term, pid, tf) VALUES (?, ?, ?)", rows_to_insert)
        cur.execute("COMMIT;")  # tmp_tf 삽입 종료

        # ───── 2) doclen 집계 & meta(N, avgdl) 기록 ─────
        cur.executescript("""
        DELETE FROM doclen;
        INSERT INTO doclen(pid, dl)
        SELECT pid, SUM(tf) AS dl
        FROM tmp_tf
        GROUP BY pid;
        """)
        conn.commit()

        N = cur.execute("SELECT COUNT(*) FROM doclen").fetchone()[0]
        avgdl_row = cur.execute("SELECT AVG(dl) FROM doclen").fetchone()
        avgdl = float(avgdl_row[0] if avgdl_row and avgdl_row[0] is not None else 0.0)

        cur.execute("DELETE FROM meta;")
        cur.executemany("INSERT INTO meta(k, v) VALUES (?, ?)", [("N", str(N)), ("avgdl", f"{avgdl:.6f}")])
        conn.commit()

        # ───── 3) postings(term→[[pid,tf]...], df) 집계 ─────
        cur.execute("DELETE FROM postings;")
        conn.commit()

        cur.execute("BEGIN;")
        term_cursor = cur.execute("SELECT DISTINCT term FROM tmp_tf")
        batch = []
        for (term,) in term_cursor:
            rows = cur.execute("SELECT pid, SUM(tf) FROM tmp_tf WHERE term = ? GROUP BY pid",
                               (term,)).fetchall()
            df = len(rows)
            plist = [[str(pid), int(tf)] for (pid, tf) in rows]  # [[pid, tf], ...]
            blob = sqlite3.Binary(zlib.compress(orjson.dumps(plist)))
            batch.append((term, int(df), blob))
            if len(batch) >= POSTINGS_BATCH_SIZE:
                cur.executemany("INSERT INTO postings(term, df, blob) VALUES (?, ?, ?)", batch)
                batch.clear()
        if batch:
            cur.executemany("INSERT INTO postings(term, df, blob) VALUES (?, ?, ?)", batch)
        cur.execute("COMMIT;")

        print(f"\n◎ SQLite index saved → {OUT_SQLITE_DB}")
        print(f"   N={N:,}, avgdl={avgdl:.4f}")
        print("\n=== Sample mapped passages ===")
        for s in sample_out:
            print(s)

if __name__=="__main__":
    main()
