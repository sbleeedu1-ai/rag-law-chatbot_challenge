"""브라우저 채팅 화면
실행:  streamlit run app.py
(먼저 python build_index.py 로 DB가 만들어져 있어야 함)
"""
import streamlit as st
from rag import RagBot

st.set_page_config(page_title="저작권법 챗봇", page_icon="⚖️")


# Streamlit 은 입력이 있을 때마다 이 파일을 처음부터 다시 실행한다.
# cache_resource 가 없으면 질문할 때마다 bge-m3 로딩 + DB 연결이 반복됨.
@st.cache_resource(show_spinner="임베딩 모델과 조문 DB를 불러오는 중...")
def get_bot() -> RagBot:
    return RagBot()


def render_sources(hits: list[dict]):
    """답변 아래 근거 조항 펼쳐보기"""
    with st.expander(f"근거 조항 {len(hits)}개 보기"):
        for h in hits:
            m = h["meta"]
            loc = " / ".join(x for x in (m["chapter"], m["section"], m["subsec"]) if x)
            st.markdown(f"**{m['article']}**  (유사도 {h['similarity']:.2f})")
            st.caption(loc)
            st.text(h["text"])


st.title("저작권법 질의응답")
st.caption("저작권법 원문에서 관련 조항을 찾아, 그 조항만 근거로 답합니다.")

try:
    bot = get_bot()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# 대화 기록도 재실행 때 사라지지 않도록 session_state 에 보관
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("검색 설정")
    top_k = st.slider("참고할 조항 수", min_value=3, max_value=10, value=5)
    exclude_buchik = st.checkbox("부칙 제외", value=True)
    if st.button("대화 지우기"):
        st.session_state.messages = []

# 지난 대화 다시 그리기
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
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
            with st.spinner("조항을 찾고 답변을 작성하는 중..."):
                answer, hits = bot.ask(question, top_k=top_k, where=where)
        except Exception as e:
            st.error(f"답변 생성에 실패했습니다: {e}")
            st.stop()

        st.markdown(answer)
        render_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )
