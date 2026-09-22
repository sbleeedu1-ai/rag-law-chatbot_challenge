"""PDF → 정제 → 청킹 → 임베딩 → Chroma 저장
문서나 청킹 규칙이 바뀔 때만 실행:  python build_index.py
"""
import os, re, time, hashlib
import numpy as np
import pymupdf
import chromadb

from common import (PDF_PATH, CACHE_DIR, DB_DIR, COLLECTION, MODEL_NAME,
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
ARTICLE = re.compile(r"^(제\d+조(?:의\d+)?\s*\([^)]*\))", re.M)
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


def parse_heading(m) -> tuple[str, str]:
    """매칭된 장/절/관 제목을 (깨끗한 이름, 태그) 로 분리"""
    raw  = f"{m.group(1)} {m.group(2)}".strip()
    tag  = TAG.search(raw)
    name = TAG.sub("", raw).strip()
    return name, (tag.group().strip() if tag else "")


def scan_heading(lines, chapter, chapter_note, section, section_note, subsec, subsec_note):
    """장/절/관/부칙 제목을 만나면 갱신. 상위 단위가 바뀌면 하위는 초기화."""
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
                "article_no": title.split("(")[0],      # "제39조(보호기간의 원칙)" → "제39조"
                "chapter": chapter, "chapter_note": chapter_note,
                "section": section, "section_note": section_note,
                "subsec": subsec, "subsec_note": subsec_note,
            })

        state = scan_heading(rest.splitlines(), *state)

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
def save_to_chroma(chunks: list[dict], texts: list[str], vecs: np.ndarray):
    assert len(texts) == len(vecs) == len(chunks)
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)     # 있으면 지우고, 없으면 넘어감
    except Exception:
        pass

    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=vecs.tolist(),
        metadatas=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
    )
    print("저장된 청크:", col.count())


def main():
    raw = "\n".join(pdf_to_pages(PDF_PATH))
    clean = clean_text(raw)
    print(f"원문 {len(raw):,}자 → 정제 후 {len(clean):,}자")

    chunks = chunk_by_article(clean)
    texts = [c["text"] for c in chunks]
    lens = [len(t) for t in texts]
    print(f"청크 {len(chunks)}개 / 평균 {np.mean(lens):.0f}자 / 최대 {max(lens)}자")

    embedder = load_embedder()
    vecs = embed_cached(embedder, "records", texts)
    save_to_chroma(chunks, texts, vecs)


if __name__ == "__main__":
    main()
