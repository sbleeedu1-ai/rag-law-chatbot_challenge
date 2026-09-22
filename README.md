# 법률 RAG 챗봇

로컬 PC에서 돌아가는 문서 기반 RAG(검색 증강 생성) 챗봇. PDF 법령 원문을 조문 단위로 쪼개 벡터 DB에 저장해두고, 질문이 들어오면 관련 조문을 찾아 **그 조문만 근거로** 답변한다. 문서에 없는 내용은 "문서에서 확인되지 않습니다"라고 답하도록 강제한다.

## 과제 배경

"도전과제 — 내 PC에서 돌아가는 문서 챗봇" (오픈채팅 스터디, slog.gg 게시글) 대응용으로 만든 프로젝트.

- **목표**: 브라우저 localhost에서, 내가 고른 문서에 질문하면 근거 조항과 함께 답하는 화면 만들기 (배포는 하지 않음, 코랩이 아닌 로컬 실행)
- **규칙**: API 키는 코드에 직접 쓰지 않고 `.env`에 저장 + 절대 GitHub에 올리지 않기 / 답변 생성은 저렴한 API 모델(`gpt-4o-mini` 등) 사용
- **체크포인트**: ① 로컬 환경 구축·API 호출 → ② 문서 하나로 RAG 구축(추출→정제→청킹→임베딩→Chroma, 근거 포함) → ③ Streamlit 채팅 화면
- **추가 과제** (이 저장소에 모두 반영됨): 대화 이력 유지 / 청킹 전략 2개 비교 / 여러 문서 필터링 검색

## 빠른 시작

```bash
conda create -n rag_challenge python=3.12 -y
conda activate rag_challenge
pip install -r requirements.txt
```

GPU(CUDA)를 쓰려면 `requirements.txt`의 `torch` 줄 대신 conda로 따로 설치한다 (pip 기본은 CPU 전용 빌드로 resolve됨):

```bash
conda install -c conda-forge pytorch-gpu
```

`.env` 파일에 API 키 저장 (형식은 `.env.example` 참고):

```
OPENAI_API_KEY=sk-...
```

인덱스 빌드 (최초 1회, 또는 문서·청킹 규칙이 바뀔 때만 재실행):

```bash
python build_index.py
```

실행:

```bash
python rag.py            # 터미널 질의응답
streamlit run app.py     # 브라우저 채팅 화면
```

## 파일 구조

```
law.pdf, labor_standards_act.pdf   원본 법령 PDF (law.go.kr, 시행예정조문 제외한 현행본)
common.py       양쪽이 공유하는 설정 + 임베딩 모델 로더 (GPU/CPU 자동 전환)
build_index.py  PDF → 정제 → 청킹(2가지 전략) → 임베딩 → Chroma 저장
rag.py          검색 + 답변 생성 (RagBot, Memory). 멀티턴, 근거 인용 검증 포함
app.py          Streamlit 화면. 문서/청킹 전략 선택, 근거 조항 표시
my_chroma_db/, rag_rules_cache/    build_index.py가 생성하는 산출물 (git 추적 안 함)
back_up/        이전 iteration 버전들 (참고용으로 남겨둠, 실행 안 됨)
```

## 파이프라인

```
PDF → pdf_to_pages() → clean_text() (머리말·쪽번호 제거) → 청킹 → 임베딩(BAAI/bge-m3) → Chroma 저장
```

### 청킹 전략 2가지 (추가과제 — 비교용)

- **article** (`chunk_by_article`): 조 → 항(①②③) → 호(1. 2. 3.) 순으로 법령 구조를 따라 분할. 조문 경계를 넘지 않음.
- **length** (`chunk_by_length`): 조·항 구조를 무시하고 글자수(기본 400자, 50자 오버랩)로 슬라이딩 윈도우 분할. `article` 대비 검색·근거 검증 정확도가 낮아지는 걸 실제로 확인할 수 있음 (아래 "알려진 한계" 참고).

두 전략 모두 같은 Chroma DB(`my_chroma_db/`) 안에 별도 컬렉션(`rules_article`, `rules_length`)으로 저장되며, `app.py` 사이드바에서 전환 가능.

### 다중 문서 (추가과제)

`common.DOCUMENTS`에 등록된 PDF마다 따로 청킹(장/절 상태가 문서 간에 섞이지 않게)한 뒤 `doc` 메타데이터를 붙여 같은 컬렉션에 합쳐 저장한다. 조문 번호가 문서 간에 겹칠 수 있어서(예: 두 법 모두 "제10조" 있음) 근거 인용에 항상 법률명을 포함시키도록 `rag.py`의 인용 파싱·검증 로직 전체가 문서명까지 함께 매칭한다.

### 검색 (`RagBot.search`)

1. 질문에 "제N조" 형태가 있으면 메타데이터로 직접 조회 (`lookup`)
2. 그 위를 벡터 검색으로 채움 (top_k)
3. 멀티턴 대화에서는 `rewrite()`가 이전 대화 맥락으로 질문을 검색용으로 재작성한 뒤 검색

### 근거 검증

LLM 답변의 "근거: ..." 줄을 파싱해서(`parse_citations`), 실제로 검색된 청크에 있는 법률·조·항인지 대조한다(`verify_citations`). 검색 결과에 없는 근거를 인용하면 화면에 경고로 표시된다.

## GPU/CPU 자동 전환

`common.load_embedder()`는 여유 VRAM이 4GB 이상이면 GPU(fp16)로, 아니면 CPU(fp32)로 임베딩 모델을 로드한다. GPU 로딩 자체가 실패해도(드라이버 문제 등) 예외를 잡아 CPU로 자동 전환한다.

Windows에서 `conda activate` 없이 `python.exe`를 직접 실행하면 cuDNN 관련 DLL을 못 찾아 죽는 문제가 있어서, `common.py`가 모듈 로드 시점에 해당 conda 환경의 `Library/bin`을 PATH에 강제로 추가한다.

## 알려진 한계

- 항(①②) 기호가 없는 조문(조문 전체가 한 덩어리인 경우)은 `verify_citations`이 어떤 항 번호를 인용해도 통과시킨다 — "조문 전체"로 간주하기 때문. 존재하지 않는 항 번호를 인용해도 못 잡을 수 있다.
- `length` 전략은 조·항 경계와 무관하게 잘리므로, 항 기호가 청크 중간에서 끊겨 위 검증 로직이 더 자주 느슨해진다 (실제 비교 시 관찰됨).
- `law.go.kr` PDF는 화면에 보이는 줄 단위로 텍스트가 추출돼 문장 중간에 줄바꿈이 섞인다. 청킹·임베딩 정확도에는 영향 없고, `app.py`의 `_display_text()`가 화면 표시용으로만 다듬는다.
