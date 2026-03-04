import os
import re
import math
import sqlite3
import zlib
import orjson
from collections import defaultdict, Counter
from contextlib import closing
from typing import List, Tuple
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

import spacy
# 🚨 [수정] spacy-entity-linker 대신 DBpedia Spotlight 라이브러리 임포트
import spacy_dbpedia_spotlight 
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

nltk.download('averaged_perceptron_tagger_eng')

# ──────────────────── 경로 & 하이퍼파라미터 ────────────────────
MSMARCO_DIR = "C:/Users/USER/Desktop/SSE/dataset/ms_marco"
PRETRAINED_MODEL = "bert-base-uncased"
CTX_CKPT_PATH = "C:/Users/USER/Desktop/SSE/save_model/best_bi_encoder_in_batch_0909.pt"
GLOSS_VEC_PATH = "C:/Users/USER/Desktop/SSE/dataset/wordnet_gloss_embeddings_0909.pt"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_SQLITE_DB = "C:/Users/USER/Desktop/SSE/dataset/0223_revised_msmarco_passage_index.sqlite"
DEBUG_SAMPLE_N = 10  
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# 오프라인 로컬 처리이므로 배치 단위로 묶어서 빠르게 밀어넣습니다.
SPACY_PIPE_BATCH = 256 
POSTINGS_BATCH_SIZE = 50_000 

if os.name == "nt":
    NUM_WORKERS = 0
    PERSISTENT_WORKERS = False


# ──────────────────── NLTK/모델 로딩 ────────────────────
HERE = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
NLTK_DIR = os.path.join(HERE, ".nltk_data")

def ensure_nltk():
    os.makedirs(NLTK_DIR, exist_ok=True)
    if NLTK_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DIR)
    
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
            nltk.download(pkg, download_dir=NLTK_DIR, quiet=True)


# ──────────────────── 전역 변수 ────────────────────
STOP = set()
nlp = None
tokenizer = None
LEMM = WordNetLemmatizer()


# ──────────────────── 텍스트 정제 함수 ────────────────────
def clean_text_for_indexing(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ──────────────────── Entity Linking (이중 그물망 - Spotlight 버전) ────────────────────
def extract_ne_from_doc(doc):
    ne_items = []
    ne_spans = []
    
    # 🚨 [수정] DBpedia Spotlight는 doc.ents에 모든 결과를 통합해서 넣어줍니다.
    linked_chars = set()

    for ent in doc.ents:
        start_char, end_char = ent.start_char, ent.end_char
        
        # 겹치는 영역 중복 방지 (가장 먼저 잡힌 긴 단어 우선)
        is_overlap = any(i in linked_chars for i in range(start_char, end_char))
        if is_overlap:
            continue
            
        # 🕸️ 1차 그물: DBpedia Spotlight 매핑 (kb_id_ 가 존재하는 경우)
        if ent.kb_id_:
            # URL (예: http://dbpedia.org/resource/United_States)에서 마지막 ID만 추출
            clean_id = ent.kb_id_.split('/')[-1]
            term = f"NE::DBPEDIA::{clean_id}"
            
        # 🕸️ 2차 그물: spaCy 기본 NER (Spotlight가 놓쳤지만 고유명사로 잡힌 경우)
        else:
            fallback_text = ent.text.lower().replace(" ", "_")
            term = f"NE::{ent.label_}::{fallback_text}"
            
        ne_items.append({
            "term": term,
            "span": (start_char, end_char)
        })
        ne_spans.append((start_char, end_char))
        linked_chars.update(range(start_char, end_char))
            
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
    
def tokenize_to_words(text):
    clean_text = text.strip()
    if not clean_text: return []
    try:
        # nlp.tokenizer()를 유지하여 파이프라인 실행 없이 순수 단어만 쪼갬
        doc = nlp.tokenizer(clean_text) 
        return [t.text for t in doc]
    except Exception as e:
        return clean_text.split()
    
def lemmatize_tokens(words: List[str]) -> Tuple[List[str], List[str]]:
    tagged = pos_tag(words)
    L = WordNetLemmatizer()
    lemmas, tags = [], []
    for w, tag in tagged:
        wp = to_wn_pos(tag)
        w_low = w.lower()
        if wp: lemmas.append(L.lemmatize(w_low, pos=wp))
        else: lemmas.append(L.lemmatize(w_low))
        tags.append(tag)
    return lemmas, tags
    
def token_in_any_span(start, end, spans):
    for s,e in spans:
        if not (end <= s or start >= e): return True
    return False


# ──────────────────── SpanContextEncoder & Dataset ────────────────────
class SpanContextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name).to(self.device)

class PassageDataset(Dataset):
    def __init__(self, collection_path, sample_limit=None):
        self.passages={}
        with open(collection_path, encoding="utf-8") as f:
            for line in f:
                try:
                    pid, txt = line.strip().split("\t", 1)
                    self.passages[pid] = txt
                except ValueError: continue
        self.pids = list(self.passages.keys())
        if sample_limit: self.pids = self.pids[:sample_limit]
            
    def __len__(self): return len(self.pids)
    def __getitem__(self, idx): return self.pids[idx], self.passages[self.pids[idx]]

def collate_fn(batch):
    pids, raw_texts = zip(*batch)
    clean_texts = [clean_text_for_indexing(t) for t in raw_texts]
    words = [tokenize_to_words(t) for t in clean_texts]
    return list(pids), words, clean_texts


# ──────────────────── WordNet 사전 & SQLite 초기화 ────────────────────
lemma2syns = defaultdict(list)
try:
    for _syn in wn.all_synsets():
        for _l in _syn.lemmas():
            lemma2syns[_l.name().lower()].append(_syn.name())
except LookupError: pass

if os.path.exists(GLOSS_VEC_PATH):
    synset2emb = torch.load(GLOSS_VEC_PATH, map_location=DEVICE)
else:
    synset2emb = {}

def init_db(db_path):
    if os.path.exists(db_path): os.remove(db_path)
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
    """)
    conn.commit()
    return conn


# ──────────────────── 메인 인덱싱 ────────────────────
def main():
    ensure_nltk()
    if not lemma2syns:
        for _syn in wn.all_synsets():
            for _l in _syn.lemmas(): lemma2syns[_l.name().lower()].append(_syn.name())

    from nltk.corpus import stopwords
    global STOP
    STOP = set(stopwords.words('english')) | {"of","the","and","in","on","for","to","a","an"}
    
    global nlp, tokenizer
    print("Loading spaCy and DBpedia Spotlight...")
    
    if torch.cuda.is_available():
        spacy.require_gpu()
        
    # 🚨 [수정] DBpedia Spotlight 파이프라인 연결
    nlp = spacy.load("en_core_web_sm", disable=["textcat", "lemmatizer"])
    nlp.add_pipe(
        'dbpedia_spotlight', 
        config={'dbpedia_rest_endpoint': 'http://localhost:2222/rest'}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, use_fast=True)
    
    collection_path = os.path.join(MSMARCO_DIR,"collection.tsv")
    
    # 테스트용 데이터 제한 
    ds = PassageDataset(collection_path, sample_limit=None) 
    
    dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, 
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    
    ctx = SpanContextEncoder(PRETRAINED_MODEL, device=DEVICE)
    if os.path.exists(CTX_CKPT_PATH):
        full_state = torch.load(CTX_CKPT_PATH, map_location=DEVICE)
        ctx_only = {k.replace("ctx_enc.encoder.","").replace("encoder.",""): v for k, v in full_state.items() if "encoder." in k}
        ctx.encoder.load_state_dict(ctx_only, strict=False)
    ctx.encoder.eval()
    
    os.makedirs(os.path.dirname(OUT_SQLITE_DB), exist_ok=True)
    
    with closing(init_db(OUT_SQLITE_DB)) as conn:
        cur = conn.cursor()
        sample_out=[]
        total_docs = 0
        
        print(f"Starting Indexing on Device: {DEVICE}")
        cur.execute("BEGIN;")
        
        with torch.no_grad():
            for pids, words_batch, texts_batch in tqdm(dl, desc="Mapping passages"):
                
                # 🟢 파이프라인 단계를 하나씩 밟으며 에러 격리
                docs = []
                for txt in texts_batch:
                    doc = nlp.make_doc(txt) 
                    for name, proc in nlp.pipeline: 
                        try:
                            doc = proc(doc)
                        except Exception:
                            # 로컬 서버와 통신 중 발생하는 일시적인 타임아웃/에러 무시
                            pass
                    docs.append(doc)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE.type=="cuda")):
                    enc = tokenizer(words_batch, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True).to(DEVICE, non_blocking=True)
                    last_hidden = ctx.encoder(**enc).last_hidden_state
                
                rows_to_insert = []
                
                for bi, (pid, doc) in enumerate(zip(pids, docs)):
                    total_docs += 1
                    words_i = words_batch[bi]
                    word_ids = enc.word_ids(batch_index=bi)
                    hidden_i = last_hidden[bi]
                    
                    word_spans = [(t.idx, t.idx+len(t.text)) for t in doc]
                    span_align_ok = (len(words_i) == len(word_spans))
                    
                    ne_items, ne_spans = extract_ne_from_doc(doc)
                    ne_terms = []
                    
                    if span_align_ok:
                        for item in ne_items:
                            term, (s, e) = item["term"], item["span"]
                            for w, (ws, we) in enumerate(word_spans):
                                if not (we <= s or ws >= e):
                                    ne_terms.append(term)
                    else:
                        ne_terms.extend([it["term"] for it in ne_items])
                        
                    lemmas, tags = lemmatize_tokens(words_i)
                    wsd_terms = []
                    seen = set()
                    unique_word_indices = [w for w in word_ids if w is not None]
                    
                    for w in dict.fromkeys(unique_word_indices):
                        if w in seen: continue
                        seen.add(w)
                        
                        if span_align_ok:
                            ws, we = word_spans[w]
                            if token_in_any_span(ws, we, ne_spans): continue
                        
                        span_mask = torch.tensor([idx==w for idx in word_ids], device=DEVICE)
                        if span_mask.sum().item() == 0: continue
                            
                        emb = hidden_i[span_mask].mean(0)
                        lemma = lemmas[w].lower()
                        cands = lemma2syns.get(lemma, [])
                        
                        if not cands: continue
                        cands = [s for s in cands if s in synset2emb]
                        if not cands: continue
                        
                        cand_embs = torch.stack([synset2emb[s] for s in cands])
                        sims = F.cosine_similarity(emb.unsqueeze(0), cand_embs)
                        best = cands[sims.argmax().item()]
                        wsd_terms.append(best)
                        
                    terms_all = ne_terms + wsd_terms
                    tf_counter = Counter(terms_all)
                    rows_to_insert.extend([(term, pid, int(tf)) for term, tf in tf_counter.items()])
                    
                    if len(sample_out) < DEBUG_SAMPLE_N:
                        sample_out.append({"pid": pid, "terms": sorted(tf_counter.keys())})
                        
                cur.executemany("INSERT INTO tmp_tf(term, pid, tf) VALUES (?, ?, ?)", rows_to_insert)
        
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
        cur.executemany("INSERT INTO meta(k, v) VALUES (?, ?)", [("N", str(N)), ("avgdl", f"{avgdl:.6f}")])
        conn.commit()
        
        print("Building Postings Table...")
        cur.execute("DELETE FROM postings;")
        conn.commit()
        
        print("\n=== 🚨 팩트 체크 ===")
        print(f"1. 준비된 문서 개수: {len(ds)}개")
        print(f"2. 로드된 WordNet 사전 단어 수: {len(synset2emb)}개")
        print("===================\n")

        cur.execute("BEGIN;")

        print(f"Starting Indexing on Device: {DEVICE}")
        term_cursor = cur.execute("SELECT DISTINCT term FROM tmp_tf")
        
        batch = []
        for (term,) in tqdm(term_cursor, desc="Postings Construction"):
            rows = cur.execute("SELECT pid, SUM(tf) FROM tmp_tf WHERE term = ? GROUP BY pid", (term,)).fetchall()
            df = len(rows)
            plist = [[str(pid), int(tf)] for (pid, tf) in rows]
            blob = sqlite3.Binary(zlib.compress(orjson.dumps(plist)))
            batch.append((term, int(df), blob))
            
            if len(batch) >= POSTINGS_BATCH_SIZE:
                cur.executemany("INSERT INTO postings(term, df, blob) VALUES (?, ?, ?)", batch)
                batch.clear()
        
        if batch:
            cur.executemany("INSERT INTO postings(term, df, blob) VALUES (?, ?, ?)", batch)
            
        cur.execute("COMMIT;")
        
        print(f"\n◎ SQLite index saved → {OUT_SQLITE_DB}")
        print(f" N={N:,}, avgdl={avgdl:.4f}")
        for s in sample_out: print(s)

if __name__=="__main__":
    main()