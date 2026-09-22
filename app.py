"""브라우저 채팅 화면 (멀티턴)
실행:  streamlit run app.py
(먼저 python build_index.py 로 DB가 만들어져 있어야 함)
"""
import re
import streamlit as st
from rag import RagBot, Memory
from common import STRATEGIES, STRATEGY_LABELS, DOCUMENTS

st.set_page_config(page_title="법률 챗봇", page_icon="⚖️")

DOC_LABELS = list(DOCUMENTS.values())

# PDF 추출은 화면에 보이는 한 줄 단위로 끊겨서, 문장 중간에 줄바꿈이 섞여 있다.
# 청킹 로직(원본 텍스트)은 그대로 두고, 화면에 보여줄 때만 문장이 끝나지 않은
# 줄바꿈을 공백으로 이어붙인다. 문장이 실제로 끝났거나(. > 」 』 】 ) 로 종료),
# 다음 줄이 새 항목(①②③, "1. ", "<개정 ...>")으로 시작하면 줄바꿈을 그대로 둔다.
_SOFT_BREAK = re.compile(r"(?<![.>」』】)])\n(?!\s*(?:[①-⑳]|\d+(?:의\d+)?\.\s|<))")


def _display_text(text: str) -> str:
    """근거 조항 원문을 화면에 보여줄 때만 다듬는다 (임베딩/청킹에는 영향 없음)."""
    return _SOFT_BREAK.sub(" ", text)


# Streamlit 은 입력이 있을 때마다 이 파일을 처음부터 다시 실행한다.
# cache_resource 로 무거운 RagBot 은 전략별로 한 번씩만 만든다 (strategy 가 캐시 키).
@st.cache_resource(show_spinner="임베딩 모델과 조문 DB를 불러오는 중...")
def get_bot(strategy: str) -> RagBot:
    return RagBot(strategy=strategy)


def render_sources(hits: list[dict]):
    """답변 아래 근거 조항 펼쳐보기"""
    with st.expander(f"근거 조항 {len(hits)}개 보기"):
        for h in hits:
            m = h["meta"]
            loc = " / ".join(x for x in (m["doc"], m["chapter"], m["section"], m["subsec"]) if x)
            score = "번호로 직접 조회" if h["similarity"] is None else f"유사도 {h['similarity']:.2f}"
            st.markdown(f"**{m['article']}**  ({score})")
            st.caption(loc)
            st.text(_display_text(h["text"]))


st.title("법률 질의응답")
st.caption("선택한 법률 원문에서 관련 조항을 찾아, 그 조항만 근거로 답합니다. 이어서 질문해도 앞 대화를 기억합니다.")

with st.sidebar:
    st.subheader("청킹 전략")
    strategy = st.radio(
        "조항을 어떤 기준으로 잘라서 검색할지 선택",
        options=list(STRATEGIES),
        format_func=lambda s: STRATEGY_LABELS[s],
        key="strategy",
    )

try:
    bot = get_bot(strategy)
except Exception as e:
    st.error(f"챗봇을 준비하는 중 오류가 발생했습니다: {e}")
    st.stop()

st.sidebar.caption(f"청크 {bot.collection.count()}개 (전략: {STRATEGY_LABELS[strategy]})")

# 사용자(브라우저 탭)마다 따로 유지되는 것들
if "messages" not in st.session_state:
    st.session_state.messages = []          # 화면에 그릴 대화
if "memory" not in st.session_state:
    st.session_state.memory = Memory()      # LLM 에 넘길 기억
if "last_strategy" not in st.session_state:
    st.session_state.last_strategy = strategy

# 전략을 바꾸면 이전 전략의 청크를 근거로 한 대화가 섞이므로 대화를 초기화
if st.session_state.last_strategy != strategy:
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.last_strategy = strategy
    st.sidebar.info("전략이 바뀌어 대화를 초기화했습니다.")

with st.sidebar:
    st.subheader("검색 설정")
    selected_docs = st.multiselect("검색할 법률", options=DOC_LABELS, default=DOC_LABELS)
    top_k = st.slider("참고할 조항 수", min_value=3, max_value=10, value=5)
    exclude_buchik = st.checkbox("부칙 제외", value=True)
    if st.button("대화 지우기"):
        st.session_state.messages = []
        st.session_state.memory.clear()

    st.subheader("지금까지 다룬 조항")
    topics = st.session_state.memory.topics
    if topics:
        for t in reversed(topics):          # 최근 것부터
            st.caption(t)
    else:
        st.caption("아직 없음")

# 지난 대화 다시 그리기
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("search_query"):
            st.caption(f"검색어: {msg['search_query']}")
        st.markdown(msg["content"])
        if msg.get("unverified"):
            st.warning("검색된 근거에서 확인되지 않는 인용: " + ", ".join(msg["unverified"]))
        if msg.get("hits"):
            render_sources(msg["hits"])

# 새 질문
if question := st.chat_input("예: 저작권은 사후 얼마나 유지되나요?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        conds = []
        if exclude_buchik:
            conds.append({"chapter": {"$ne": "부칙"}})
        if selected_docs and len(selected_docs) < len(DOC_LABELS):
            conds.append({"doc": {"$in": selected_docs}})
        where = conds[0] if len(conds) == 1 else ({"$and": conds} if conds else None)
        if not selected_docs:
            st.warning("검색할 법률을 하나 이상 선택해주세요.")
            st.stop()
        try:
            with st.spinner("질문을 해석하고 조항을 찾는 중..."):
                result = bot.ask(question, memory=st.session_state.memory,
                                 top_k=top_k, where=where)
        except Exception as e:
            st.error(f"답변 생성에 실패했습니다: {e}")
            st.stop()

        rewritten = result["search_query"] if result["search_query"] != question else ""
        if rewritten:
            st.caption(f"검색어: {rewritten}")
        st.markdown(result["answer"])
        if result["unverified"]:
            st.warning("검색된 근거에서 확인되지 않는 인용: " + ", ".join(result["unverified"]))
        render_sources(result["hits"])

    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"],
        "hits": result["hits"], "search_query": rewritten,
        "unverified": result["unverified"],
    })
    st.rerun()   # 사이드바의 '다룬 조항'을 방금 턴까지 반영해서 다시 그림
