"""build_index.py 와 rag.py 가 함께 쓰는 설정과 임베딩 함수"""
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# 경로는 이 파일 위치 기준 → 어느 폴더에서 실행해도 같은 곳을 가리킴
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PDF_PATH   = os.path.join(BASE_DIR, "law.pdf")
CACHE_DIR  = os.path.join(BASE_DIR, "rag_rules_cache")
DB_DIR     = os.path.join(BASE_DIR, "my_chroma_db")
COLLECTION = "rules"

MODEL_NAME  = "BAAI/bge-m3"
MAX_SEQ_LEN = 1024


def load_embedder():
    """임베딩 모델 로딩 (무거움 → 호출하는 쪽에서 한 번만 부를 것)"""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = MAX_SEQ_LEN
    return model


def embed_texts(embedder, texts: list[str], batch: int = 32) -> np.ndarray:
    """정규화된 임베딩 → 내적 = 코사인 유사도"""
    return embedder.encode(
        texts, batch_size=batch, normalize_embeddings=True,
        show_progress_bar=len(texts) > 50, convert_to_numpy=True,
    ).astype(np.float32)
