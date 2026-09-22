"""PDF → 정제 → 청킹 → 임베딩 → Chroma 저장
문서나 청킹 규칙이 바뀔 때만 실행:  python build_index.py
"""
import os, re, time, hashlib
import numpy as np
import pymupdf
import chromadb

from common import (LAW_DIR, CACHE_DIR, DB_DIR, STRATEGIES, DOCUMENTS, MODEL_NAME,
                    load_embedder, embed_texts)


# ── 1. 추출 ─────────────────────────────────────────────
def pdf_to_pages(path: str) -> list[str]:
    """페이지별 텍스트 리스트로 반환"""
    with pymupdf.open(path) as doc:
        return [page.get_text() for page in doc]


# ── 2. 정제 ─────────────────────────────────────────────
FOOTER = re.compile(r"법제처\s*\d*\s*국가법령정보센터")

def clean_text(raw: str, law_name: str = "저작권법") -> str:
    raw = FOOTER.sub("", raw)
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
        out.append(s)
    return "\n".join(out)


# ── 3. 청킹 ─────────────────────────────────────────────
ARTICLE = re.compile(r"^(제\d+조(?:의\d+)?\s*(?:\([^)]*\)|삭제))", re.M)
ARTICLE_NO = re.compile(r"^제\d+조(?:의\d+)?")   # 조문 제목에서 순수 조 번호만 뽑을 때 사용
CHAPTER = re.compile(r"^\s*(제\s*\d+\s*장(?:의\s*\d+)?)\s*(.*)$")
SECTION = re.compile(r"^\s*(제\s*\d+\s*절(?:의\s*\d+)?)\s*(.*)$")
SUBSEC  = re.compile(r"^\s*(제\s*\d+\s*관(?:의\s*\d+)?)\s*(.*)$")
BUCHIK  = re.compile(r"^\s*부\s*칙\s*(.*)$")
TAG     = re.compile(r"\s*<[^>]*>")
PARA    = re.compile(r"(?=[①-⑳])")
HO      = re.compile(r"(?=^\s*\d+(?:의\d+)?\.\s)", re.M)


def is_heading(line: str) -> bool:
    return bool(BUCHIK.match(line) or CHAPTER.match(line)
                or SECTION.match(line) or SUBSEC.match(line))


def parse_heading(m) -> str:
    """매칭된 장/절/관 제목에서 <개정 ...> 같은 태그를 떼어낸 이름만 반환"""
    raw = f"{m.group(1)} {m.group(2)}".strip()
    return TAG.sub("", raw).strip()


def scan_heading(lines, chapter, section, subsec):
    """장/절/관/부칙 제목을 만나면 갱신. 상위 단위가 바뀌면 하위는 초기화."""
    for line in lines:
        if BUCHIK.match(line):
            chapter, section, subsec = "부칙", "", ""
            continue
        m = CHAPTER.match(line)
        if m:
            chapter, section, subsec = parse_heading(m), "", ""
            continue
        m = SECTION.match(line)
        if m:
            section, subsec = parse_heading(m), ""
            continue
        m = SUBSEC.match(line)
        if m:
            subsec = parse_heading(m)
    return chapter, section, subsec


def split_items(title: str, text: str, max_len: int = 500) -> list[str]:
    """호(1. 2. 3.) 단위 분할. 짧은 호는 max_len까지 병합.
    맨 앞(항 머리말, ①... 등 첫 호 이전 부분)을 모든 조각 앞에 반복해서 붙인다 —
    안 그러면 뒤로 밀려 나뉜 조각은 자기가 몇 항 소속인지(①인지 ②인지) 텍스트만
    봐서는 알 수 없게 된다."""
    parts = [p.strip() for p in HO.split(text) if p.strip()]
    if len(parts) <= 1:
        return [text]

    head, items = parts[0], parts[1:]
    out, buf = [], head
    for p in items:
        candidate = f"{buf}\n{p}"
        if len(candidate) <= max_len:
            buf = candidate
        else:
            out.append(buf)
            buf = f"{head}\n{p}"   # 새 조각도 항 머리말부터 다시 시작 (① 유지)
    out.append(buf)
    return [c if c.startswith(title) else f"{title}\n{c}" for c in out]


def split_paragraphs(title: str, body: str, max_len: int = 500) -> list[str]:
    """조문을 항(①②③) 단위로 분할. 항이 없으면 조문 통째로. 긴 조각은 호 단위로."""
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
    state = scan_heading(pieces[0].splitlines(), "", "", "")

    chunks, seen = [], set()
    for k in range(1, len(pieces), 2):
        title, rest = pieces[k].strip(), pieces[k + 1]
        body_lines = [l for l in rest.splitlines() if not is_heading(l)]
        body = (title + "\n".join(body_lines)).strip()

        if body in seen:
            continue
        seen.add(body)

        chapter, section, subsec = state
        for c in split_paragraphs(title, body):
            chunks.append({
                "text": c, "article": title,
                "article_no": ARTICLE_NO.match(title).group(),   # "제39조(보호기간의 원칙)" / "제35조 삭제" → "제39조" / "제35조"
                "chapter": chapter, "section": section, "subsec": subsec,
            })

        state = scan_heading(rest.splitlines(), *state)

    return chunks


# ── 3-2. 청킹 (비교용 대안 전략) ─────────────────────────
def chunk_by_length(text: str, chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """조·항 구조를 무시하고 글자수로 자르는 슬라이딩 윈도우 청킹.
    chunk_by_article 과 달리 조문 경계와 상관없이 잘리므로, 청크가 문장 중간에서
    끊기거나 여러 조문이 한 청크에 섞일 수 있다. (구조 기반 청킹과 비교용)
    메타데이터(장/절/관/조문)는 그 청크가 시작하는 지점 기준으로 채운다."""
    lines = text.splitlines()
    chapter = section = subsec = ""
    cur_article, cur_article_no = "", ""

    chunks: list[dict] = []
    buf: list[str] = []
    buf_len = 0

    def flush(carry_overlap: bool):
        nonlocal buf, buf_len
        chunk_text = "\n".join(buf).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text, "article": cur_article, "article_no": cur_article_no,
                "chapter": chapter, "section": section, "subsec": subsec,
            })
        if carry_overlap and overlap > 0 and chunk_text:
            tail = chunk_text[-overlap:]
            buf, buf_len = [tail], len(tail)
        else:
            buf, buf_len = [], 0

    for line in lines:
        m = ARTICLE.match(line)
        if m:
            cur_article = m.group(1).strip()
            cur_article_no = ARTICLE_NO.match(cur_article).group()
        if is_heading(line):
            chapter, section, subsec = scan_heading([line], chapter, section, subsec)
            continue   # 제목 줄 자체는 청크 본문에 넣지 않음 (구조 기반 청킹과 동일 처리)

        buf.append(line)
        buf_len += len(line) + 1
        if buf_len >= chunk_size:
            flush(carry_overlap=True)

    flush(carry_overlap=False)
    return chunks


# ── 4. 임베딩 캐시 ───────────────────────────────────────
def embed_cached(embedder, name: str, texts: list[str]) -> np.ndarray:
    """텍스트·모델이 같으면 저장된 임베딩 재사용"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    sig = f"{MODEL_NAME}|{embedder.max_seq_length}\u241e" + "\u241e".join(texts)
    key = hashlib.md5(sig.encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"{name}_{key}.npy")
    if os.path.exists(path):
        print(f"[캐시] {name} ({len(texts)}개)")
        return np.load(path)
    t = time.time()
    vecs = embed_texts(embedder, texts)
    np.save(path, vecs)
    print(f"[임베딩] {name} ({len(texts)}개) {time.time() - t:.1f}초")
    return vecs


# ── 5. Chroma 저장 ───────────────────────────────────────
def save_to_chroma(chunks: list[dict], texts: list[str], vecs: np.ndarray, collection: str):
    assert len(texts) == len(vecs) == len(chunks)
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(collection)     # 있으면 지우고, 없으면 넘어감
    except Exception:
        pass

    col = client.create_collection(collection, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=vecs.tolist(),
        metadatas=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
    )
    print(f"[{collection}] 저장된 청크:", col.count())


def build_strategy(strategy: str, chunk_fn, docs: list[tuple[str, str]], embedder):
    """docs: [(문서 표시명, 정제된 본문), ...]. 문서마다 따로 청킹(장/절 상태가 섞이지 않게)
    한 뒤 doc 필드를 붙여 하나로 합치고, 한 컬렉션에 함께 저장한다."""
    chunks = []
    for doc_label, clean in docs:
        doc_chunks = chunk_fn(clean)
        for c in doc_chunks:
            c["doc"] = doc_label
        chunks.extend(doc_chunks)

    texts = [c["text"] for c in chunks]
    lens = [len(t) for t in texts]
    print(f"[{strategy}] 청크 {len(chunks)}개 / 평균 {np.mean(lens):.0f}자 / 최대 {max(lens)}자")

    vecs = embed_cached(embedder, f"records_{strategy}", texts)
    save_to_chroma(chunks, texts, vecs, STRATEGIES[strategy])


def main():
    docs = []
    for filename, label in DOCUMENTS.items():
        path = os.path.join(LAW_DIR, filename)
        raw = "\n".join(pdf_to_pages(path))
        clean = clean_text(raw, law_name=label)
        print(f"[{label}] 원문 {len(raw):,}자 → 정제 후 {len(clean):,}자")
        docs.append((label, clean))

    embedder = load_embedder()
    build_strategy("article", chunk_by_article, docs, embedder)
    build_strategy("length", chunk_by_length, docs, embedder)


if __name__ == "__main__":
    main()
