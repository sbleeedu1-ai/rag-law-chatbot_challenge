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

# 청킹 전략별 컬렉션 이름 + 화면에 보여줄 이름
STRATEGIES = {
    "article": "rules_article",
    "length":  "rules_length",
}
STRATEGY_LABELS = {
    "article": "조·항 단위 (구조 기반)",
    "length":  "글자수 단위 (슬라이딩 윈도우)",
}

MODEL_NAME  = "BAAI/bge-m3"
MAX_SEQ_LEN = 1024


MIN_GPU_VRAM_GB = 4  # bge-m3 fp16 추론 권장 여유 VRAM. 미만이면 GPU 시도 자체를 안 함


def _try_load_on_gpu(model_name: str):
    """여유 VRAM이 충분할 때만 GPU 로딩을 시도하고, 실패하면 None을 반환한다.
    구형/내장 GPU(예: 2GB급)는 여기서 걸러지고, 그 외 실패(드라이버 이슈 등)는
    워밍업 인코딩에서 예외로 잡아 CPU로 넘어간다."""
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        return None

    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < MIN_GPU_VRAM_GB:
        print(f"[임베더] GPU 여유 VRAM {free_gb:.1f}GB < {MIN_GPU_VRAM_GB}GB → CPU 사용")
        return None

    try:
        model = SentenceTransformer(model_name, device="cuda")
        model.half()
        model.encode(["워밍업"], convert_to_numpy=True)  # 실제로 동작하는지 검증
        print(f"[임베더] GPU 사용 (여유 VRAM {free_gb:.1f}GB)")
        return model
    except Exception as e:
        print(f"[임베더] GPU 로딩 실패({e}) → CPU로 전환")
        return None


def load_embedder():
    """임베딩 모델 로딩 (무거움 → 호출하는 쪽에서 한 번만 부를 것)
    GPU가 있고 여유 VRAM이 충분하면 GPU(fp16)를, 아니면 CPU(fp32)를 쓴다."""
    import torch
    from sentence_transformers import SentenceTransformer

    model = _try_load_on_gpu(MODEL_NAME)
    if model is None:
        torch.set_num_threads(4)
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("[임베더] CPU 사용")

    model.max_seq_length = MAX_SEQ_LEN
    return model


def embed_texts(embedder, texts: list[str], batch: int = 32) -> np.ndarray:
    """정규화된 임베딩 → 내적 = 코사인 유사도"""
    return embedder.encode(
        texts, batch_size=batch, normalize_embeddings=True,
        show_progress_bar=len(texts) > 50, convert_to_numpy=True,
    ).astype(np.float32)
