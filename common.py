"""build_index.py 와 rag.py 가 함께 쓰는 설정과 임베딩 함수"""
import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# conda activate 없이 python.exe를 직접 실행하면 Library/bin(cuDNN 등 CUDA
# 의존 DLL이 있는 곳)이 PATH에 없어서 torch가 GPU를 잡고도 cuDNN 심볼을 못
# 찾아 죽는다. cuDNN의 하위 DLL들은 os.add_dll_directory가 아니라 실제
# PATH 환경변수를 보고 서로를 찾으므로, PATH 자체에 직접 넣어줘야 한다.
if sys.platform == "win32":
    _lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(_lib_bin) and _lib_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _lib_bin + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(_lib_bin)

# 경로는 이 파일 위치 기준 → 어느 폴더에서 실행해도 같은 곳을 가리킴
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LAW_DIR    = os.path.join(BASE_DIR, "data", "laws")
CACHE_DIR  = os.path.join(BASE_DIR, "data", "cache")
DB_DIR     = os.path.join(BASE_DIR, "data", "index")


def active_db_dir() -> str:
    active = os.path.join(DB_DIR, "active.txt")
    if not os.path.exists(active):
        return DB_DIR
    with open(active, encoding="utf-8") as f:
        build_id = f.read().strip()
    return os.path.join(DB_DIR, "builds", build_id)

# 청킹 전략별 컬렉션 이름 + 화면에 보여줄 이름
STRATEGIES = {
    "article": "rules_article",
    "length":  "rules_length",
}
STRATEGY_LABELS = {
    "article": "조·항 단위 (구조 기반)",
    "length":  "글자수 단위 (슬라이딩 윈도우)",
}

# 문서 파일명 -> 화면/근거 인용에 쓸 표시 이름
DOCUMENTS = {
    "law.pdf":                 "저작권법",
    "labor_standards_act.pdf": "근로기준법",
    "constitution.pdf": "대한민국헌법",
    "copyright_enforcement_decree.pdf": "저작권법 시행령",
    "information_network_act.pdf": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
}
LENGTH_DOCUMENT = "labor_standards_act.pdf"

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
