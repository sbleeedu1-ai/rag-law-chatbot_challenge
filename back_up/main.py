# API 호출 테스트용
import os
from dotenv import load_dotenv
from openai import OpenAI

# 텍스트 추출 정제 청킹용
import re, struct, zlib
import numpy as np
import pymupdf

# 임베딩 DB 저장용
import os, hashlib, time
import torch, chromadb
from collections import Counter
from sentence_transformers import SentenceTransformer



# PDF 파일 텍스트 추출 후 정제

pdf_path = "law.pdf"

def pdf_to_pages(path: str) -> list[str]:
    """페이지별 텍스트 리스트로 반환"""
    with pymupdf.open(path) as doc:
        return [page.get_text() for page in doc]

# 원본
# doc = pymupdf.open(pdf_path)

# def pdf_to_txt(text):
#     print("--- Extracting text ---")
#     pdf_text = []
#     for page in text:
#         pdf_text.append(page.get_text())
#     doc.close()
#     print("--- Done. ---")
#     return "\n".join(pdf_text)

FOOTER = re.compile(r"법제처\s*\d*\s*국가법령정보센터")

def clean_text(raw: str, law_name: str = "저작권법") -> str:
    raw = FOOTER.sub("", raw)          # 한 줄로 붙은 경우 먼저 제거
    drop = {"법제처", "국가법령정보센터", law_name}
    out, prev_dropped = [], False
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s in drop:
            prev_dropped = True
            continue
        if prev_dropped and re.fullmatch(r"\d{1,4}", s):   # 꼬리말 사이 쪽번호
            prev_dropped = False
            continue
        prev_dropped = False
        out.append(s)                  # ← 들여쓰기까지 제거
    return "\n".join(out)

# 원본
# def clean_text(raw: str) -> str:
#     out = []
#     for line in raw.splitlines():
#         s = line.strip()
#         if not s:
#             continue
#         if s in ["법제처", "저작권법", "국가법령정보센터"]:
#             continue
#         if re.fullmatch(r"법제처\s+\d+\s+국가법령정보센터", s):
#             continue
#         out.append(line.rstrip())
#     return "\n".join(out)

pages = pdf_to_pages(pdf_path)
raw = "\n".join(pages)
# raw = pdf_to_txt(doc) # 원본
clean = clean_text(raw)
# print(raw[:4000])
# print(clean[:4000])

# 줄어든 글자수 체크용
# print(f"원문 {len(raw):,}자 → 정제 후 {len(clean):,}자")


# 청킹

ARTICLE = re.compile(r"^(제\d+조(?:의\d+)?\s*\([^)]*\))", re.M)
CHAPTER = re.compile(r"^\s*(제\s*\d+\s*장(?:의\s*\d+)?)\s*(.*)$")
SECTION = re.compile(r"^\s*(제\s*\d+\s*절(?:의\s*\d+)?)\s*(.*)$")
SUBSEC  = re.compile(r"^\s*(제\s*\d+\s*관(?:의\s*\d+)?)\s*(.*)$")
TAG     = re.compile(r"\s*<[^>]*>")
PARA    = re.compile(r"(?=[①-⑳])")
HO      = re.compile(r"(?=^\s*\d+(?:의\d+)?\.\s)", re.M)
BUCHIK  = re.compile(r"^\s*부\s*칙\s*(.*)$")
# 원본
# ARTICLE = re.compile(r"^(제\d+조(?:의\d+)?\s*\([^)]*\))", re.M)
# CHAPTER = re.compile(r"^\s*(제\s*\d+\s*장)\s*(.*)$")


def is_heading(line: str) -> bool:
    return bool(BUCHIK.match(line) or CHAPTER.match(line) or SECTION.match(line) or SUBSEC.match(line))

def split_items(title: str, text: str, max_len: int = 500) -> list[str]:
    """호(1. 2. 3.) 단위 분할. 짧은 호는 max_len까지 병합."""
    parts = [p.strip() for p in HO.split(text) if p.strip()]
    if len(parts) <= 1:
        return [text]

    out, buf = [], parts[0]
    for p in parts[1:]:
        if len(buf) + len(p) + 1 <= max_len:
            buf = f"{buf}\n{p}"
        else:
            out.append(buf)
            buf = p
    out.append(buf)
    return [c if c.startswith(title) else f"{title}\n{c}" for c in out]

# 원본

# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
#     """글자 수 기준 고정 청킹"""
#     chunks, start = [], 0
#     while start < len(text):
#         chunk = text[start:start + chunk_size]
#         if chunk.strip():
#             chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks

def parse_heading(m) -> tuple[str, str]:
    """매칭된 장/절 제목을 (깨끗한 이름, 태그) 로 분리"""
    raw  = f"{m.group(1)} {m.group(2)}".strip()
    tag  = TAG.search(raw)
    name = TAG.sub("", raw).strip()
    return name, (tag.group().strip() if tag else "")

def scan_heading(lines, chapter, chapter_note, section, section_note, subsec, subsec_note):
    for line in lines:
        m = BUCHIK.match(line)
        if m:
            chapter, chapter_note = "부칙", m.group(1).strip()
            section = section_note = subsec = subsec_note = ""
            continue
        
        m = CHAPTER.match(line)
        if m:
            chapter, chapter_note = parse_heading(m)
            section = section_note = subsec = subsec_note = ""
            continue

        m = SECTION.match(line)
        if m:
            section, section_note = parse_heading(m)
            subsec = subsec_note = ""
            continue

        m = SUBSEC.match(line)
        if m:
            subsec, subsec_note = parse_heading(m)

    return chapter, chapter_note, section, section_note, subsec, subsec_note

def split_paragraphs(title: str, body: str, max_len: int = 500) -> list[str]:
    """조문을 항(①②③) 단위로 분할. 항이 없으면 조문 통째로."""
    parts = [p.strip() for p in PARA.split(body) if p.strip()]
    if len(parts) <= 1:
        pieces = [body]
    else:
        head, paras = parts[0], parts[1:]
        pieces = [f"{head}\n{paras[0]}"] + [f"{title}\n{p}" for p in paras[1:]]

    out = []
    for c in pieces:
        out.extend([c] if len(c) <= max_len else split_items(title, c, max_len))
    return out

def chunk_by_article(text: str) -> list[dict]:
    pieces = ARTICLE.split(text)
    state = scan_heading(pieces[0].splitlines(), "", "", "", "", "", "")

    chunks, seen = [], set()
    for k in range(1, len(pieces), 2):
        title, rest = pieces[k].strip(), pieces[k + 1]
        body_lines = [l for l in rest.splitlines() if not is_heading(l)]
        body = (title + "\n".join(body_lines)).strip()

        if body in seen:
            continue
        seen.add(body)

        chapter, chapter_note, section, section_note, subsec, subsec_note = state
        for c in split_paragraphs(title, body):
            chunks.append({
                "text": c, "article": title,
                "chapter": chapter, "chapter_note": chapter_note,
                "section": section, "section_note": section_note,
                "subsec": subsec, "subsec_note": subsec_note,
            })

        state = scan_heading(rest.splitlines(), *state)

    return chunks

# 원본
# def chunk_by_article(text: str, max_len: int = 800) -> list[str]:
#     """'제N조(제목)' 단위 청킹. 긴 조항은 제목을 붙여 다시 자른다."""
#     pieces = ARTICLE.split(text)
#     chunks, seen = [], set()
#     for k in range(1, len(pieces), 2):
#         title = pieces[k].strip()
#         body = (pieces[k] + pieces[k + 1]).strip()
#         if len(body) < 20 or body in seen:
#             continue
#         seen.add(body)
#         if len(body) <= max_len:
#             chunks.append(body)
#         else:
#             for p in chunk_text(body, max_len, 100):
#                 chunks.append(p if p.startswith(title) else f"{title} (계속)\n{p}")
#     return chunks


chunks = chunk_by_article(clean)
texts  = [c["text"] for c in chunks]

# 원본
# chunks = chunk_by_article(clean)
# print(f"{"article 단위":12s} 청크 {len(chunks):4d}개 / 평균 {int(np.mean([len(c) for c in chunks])):4d}자")

# 실제 결과 체크용

# import random

# def show_chunks(article_prefix: str = None, n: int = 5):
#     """조문 지정 시 해당 청크 전부, 없으면 무작위 n개"""
#     pool = [c for c in chunks if c["article"].startswith(article_prefix)] if article_prefix \
#            else random.sample(chunks, n)
#     for c in pool:
#         meta = " / ".join(x for x in (c["chapter"], c["section"], c["subsec"]) if x)
#         print(f"[{meta}] {c['article']} ({len(c['text'])}자)")
#         print(c["text"])
#         print("-" * 60)

# show_chunks("제39조(")   # 특정 조문
# show_chunks()            # 무작위 5개



# 임베딩과 DB

CACHE_DIR = r"C:\Users\201\Desktop\challenge_project\rag_rules_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
torch.set_num_threads(4)
# print("사용 장치: CPU")

MODEL_NAME = "BAAI/bge-m3"
embedder = SentenceTransformer(MODEL_NAME, device="cpu")
embedder.max_seq_length = 1024

# 원본
# embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
# embedder.max_seq_length = 1024


def embed_texts(texts: list[str], batch: int = 32) -> np.ndarray:
    """정규화된 임베딩 → 내적 = 코사인 유사도"""
    return embedder.encode(
        texts, batch_size=batch, normalize_embeddings=True,
        show_progress_bar=len(texts) > 50, convert_to_numpy=True,
    ).astype(np.float32)

def embed_cached(name: str, texts: list[str]) -> np.ndarray:
    """텍스트·모델이 같으면 저장된 임베딩 재사용"""
    sig = f"{MODEL_NAME}|{embedder.max_seq_length}\u241e" + "\u241e".join(texts)
    key = hashlib.md5(sig.encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"{name}_{key}.npy")
    if os.path.exists(path):
        # print(f"[캐시] {name} ({len(texts)}개)")
        return np.load(path)
    t = time.time()
    vecs = embed_texts(texts)
    np.save(path, vecs)
    # print(f"[임베딩] {name} ({len(texts)}개) {time.time() - t:.1f}초")
    return vecs

# 원본
# def embed_cached(name: str, texts: list[str]) -> np.ndarray:
#     """청크 내용이 같으면 드라이브에 저장된 임베딩을 재사용"""
#     key = hashlib.md5("\u241e".join(texts).encode()).hexdigest()[:10]
#     path = f"{CACHE_DIR}/{name}_{key}.npy"
#     if os.path.exists(path):
#         print(f"[캐시] {name} ({len(texts)}개)")
#         return np.load(path)
#     t = time.time()
#     vecs = embed_texts(texts)
#     np.save(path, vecs)
#     print(f"[임베딩] {name} ({len(texts)}개) {time.time() - t:.1f}초")
#     return vecs

chunk_embeddings = embed_cached("records", texts)

DB_DIR = r"C:\Users\201\Desktop\challenge_project\my_chroma_db"
db = chromadb.PersistentClient(path=DB_DIR)

try:
    db.delete_collection("rules")      # 있으면 지우고, 없으면 넘어감
except Exception:
    pass

collection = db.create_collection("rules", metadata={"hnsw:space": "cosine"})

assert len(texts) == len(chunk_embeddings) == len(chunks)
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=texts,
    embeddings=chunk_embeddings.tolist(),
    metadatas=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
)
# print("저장된 청크:", collection.count())

# 임베딩 / 메타데이터 체크용
# print(collection.get(ids=["chunk_0"], include=["documents", "metadatas"]))

# 원본
# client = chromadb.PersistentClient(path="my_chroma_db")

# if "rules" in [c.name for c in client.list_collections()]:
#     client.delete_collection("rules")
#     collection = client.create_collection("rules", metadata={"hnsw:space": "cosine"})
#     collection.add(
#     ids=[f"chunk_{i}" for i in range(len(chunks))],
#     documents=chunks,
#     embeddings=chunk_embeddings.tolist(),
#     metadatas=[{k: r[k] for k in ("part", "chapter", "article")} for r in chunks],
# )
# print("저장된 청크:", collection.count())


# 만든 백터 검색

def search_vectordb(query: str, top_k: int = 3, where: dict | None = None):
    """cosine 공간이므로 유사도 = 1 - distance"""
    q_emb = embed_texts([query])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
    return [
        {"id": i, "text": t, "meta": m, "similarity": 1 - d}
        for i, t, m, d in zip(res["ids"][0], res["documents"][0],
                              res["metadatas"][0], res["distances"][0])
    ]

# 원본
# def search_vectordb(query: str, top_k: int = 3):
#     """cosine 공간이므로 유사도 = 1 - distance"""
#     q_emb = embed_texts([query])[0].tolist()
#     res = collection.query(query_embeddings=[q_emb], n_results=top_k)
#     return [
#         {"id": i, "text": t, "similarity": 1 - d}
#         for i, t, d in zip(res["ids"][0], res["documents"][0], res["distances"][0])
#     ]

def show_search(query: str, top_k: int = 3, where: dict | None = None):
    print(f"=== 질문: {query} ===")
    for rank, r in enumerate(search_vectordb(query, top_k, where), 1):
        m = r["meta"]
        loc = " / ".join(x for x in (m["chapter"], m["section"], m["subsec"]) if x)
        print(f"\n[{rank}위] 유사도 {r['similarity']:.4f}  [{loc}] {m['article']}")
        print(r["text"][:150] + ("..." if len(r["text"]) > 150 else ""))


# 원본
# def show_search(query: str, top_k: int = 3):
#     print(f"=== 질문: {query} ===")
#     for rank, r in enumerate(search_vectordb(query, top_k), 1):
#         print(f"\n[{rank}위] 유사도 {r['similarity']:.4f} ")
#         print(r["text"][:150] + ("..." if len(r["text"]) > 150 else ""))

# 질문 유사도 순위 확인용
# show_search("저작권은 사후 얼마나 유지되나요?")
# show_search("저작권은 사후 얼마나 유지되나요?", where={"chapter": {"$ne": "부칙"}})


# cosine 적용 여부 체크
# r = search_vectordb("저작권은 사후 얼마나 유지되나요?", 1)[0]
# idx = int(r["id"].split("_")[1])
# q = embed_texts(["저작권은 사후 얼마나 유지되나요?"])[0]
# print(r["similarity"], float(q @ chunk_embeddings[idx]))


# api키 env 로 저장한것 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, ".env에서 OPENAI_API_KEY를 못 읽었습니다"

# API 호출
llm = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """당신은 저작권법 안내 챗봇입니다.
- 반드시 아래 [근거] 내용만 근거로 답하세요.
- 조문의 표현을 최대한 그대로 사용하며, 조문에 없는 계산·요약·추론을 덧붙이지 마세요.
- 답변 끝에 근거 조항 번호(예: 제39조 제1항)를 적으세요.
- 근거에 없는 내용이면 '문서에서 확인되지 않습니다.'라고 답하세요.
- <개정 ...>, <신설 ...> 같은 개정 이력 표기는 답변에 옮기지 않습니다. 단, 개정·신설 시기를 묻는 질문에는 이를 근거로 답합니다."""


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        loc = " / ".join(x for x in (m["chapter"], m["section"], m["subsec"]) if x)
        blocks.append(f"[{i}] ({loc})\n{h['text']}")
    return "\n\n".join(blocks)


def ask(query: str, top_k: int = 5, where: dict | None = None, show_context: bool = True) -> str:
    hits = search_vectordb(query, top_k, where)
    context = build_context(hits)
    if show_context:
        print("[검색된 조항] " + ", ".join(h["meta"]["article"] for h in hits))

    t = time.time()
    res = llm.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[근거]\n{context}\n\n[질문]\n{query}"},
        ],
    )
    if show_context:
        print(f"[생성 {time.time() - t:.1f}초]")
    return res.choices[0].message.content


print(ask("저작권은 사후 얼마나 유지되나요?"))       # 태그 없이 제39조
print(ask("제39조는 언제 개정됐나요?"))             # 2011. 6. 30.
print(ask("특허권 존속기간은 몇 년인가요?"))         # 거절
