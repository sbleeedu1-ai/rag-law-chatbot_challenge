"""브라우저 채팅 화면 (멀티턴)
실행:  streamlit run app.py
(먼저 python build_index.py 로 DB가 만들어져 있어야 함)
"""
import streamlit as st
from rag import RagBot, Memory

st.set_page_config(page_title="저작권법 챗봇", page_icon="⚖️")


# Streamlit 은 입력이 있을 때마다 이 파일을 처음부터 다시 실행한다.
# cache_resource 로 무거운 RagBot 은 서버 전체에서 한 번만 만든다.
@st.cache_resource(show_spinner="임베딩 모델과 조문 DB를 불러오는 중...")
def get_bot() -> RagBot:
    return RagBot()


def render_sources(hits: list[dict]):
    """답변 아래 근거 조항 펼쳐보기"""
    with st.expander(f"근거 조항 {len(hits)}개 보기"):
        for h in hits:
            m = h["meta"]
            loc = " / ".join(x for x in (m["chapter"], m["section"], m["subsec"]) if x)
            score = "번호로 직접 조회" if h["similarity"] is None else f"유사도 {h['similarity']:.2f}"
            st.markdown(f"**{m['article']}**  ({score})")
            st.caption(loc)
            st.text(h["text"])


st.title("저작권법 질의응답")
st.caption("저작권법 원문에서 관련 조항을 찾아, 그 조항만 근거로 답합니다. 이어서 질문해도 앞 대화를 기억합니다.")

try:
    bot = get_bot()
except Exception as e:
    st.error(f"챗봇을 준비하는 중 오류가 발생했습니다: {e}")
    st.stop()

# 사용자(브라우저 탭)마다 따로 유지되는 것들
if "messages" not in st.session_state:
    st.session_state.messages = []          # 화면에 그릴 대화
if "memory" not in st.session_state:
    st.session_state.memory = Memory()      # LLM 에 넘길 기억

with st.sidebar:
    st.subheader("검색 설정")
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
        where = {"chapter": {"$ne": "부칙"}} if exclude_buchik else None
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
