"""검색 + 답변 생성
터미널 질의응답:  python rag.py
Streamlit 에서는 RagBot 을 불러다 씀
"""
import os, time
import chromadb
from openai import OpenAI

from common import DB_DIR, COLLECTION, load_embedder, embed_texts

LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """당신은 저작권법 안내 챗봇입니다.
- 반드시 아래 [근거] 내용만 근거로 답하세요.
- 조문의 표현을 최대한 그대로 사용하며, 조문에 없는 계산·요약·추론을 덧붙이지 마세요.
- 근거에 없는 내용이면 '문서에서 확인되지 않습니다.'라고만 답하세요.
- <개정 ...>, <신설 ...> 같은 개정 이력 표기는 답변에 옮기지 마세요. 단, 개정·신설 시기를 묻는 질문에는 이를 근거로 답하세요.
- 답변 본문 다음 줄에, 아래 형식 그대로 근거를 한 줄로 적으세요. 괄호나 다른 문구를 붙이지 마세요.
  근거: 제39조 제1항
  근거: 제48조 제1항, 제2항
  근거: 제10조, 제16조 제1항"""


class RagBot:
    """무거운 자원(LLM 클라이언트, 임베딩 모델, DB)을 한 번 로딩해서 들고 있는 객체"""

    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(".env 에서 OPENAI_API_KEY 를 못 읽었습니다")
        self.llm = OpenAI(api_key=key)
        self.embedder = load_embedder()

        client = chromadb.PersistentClient(path=DB_DIR)
        try:
            self.collection = client.get_collection(COLLECTION)   # 열기만, 지우지 않음
        except Exception as e:
            raise RuntimeError("컬렉션이 없습니다. 먼저 python build_index.py 를 실행하세요") from e

    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
        """cosine 공간이므로 유사도 = 1 - distance"""
        q_emb = embed_texts(self.embedder, [query])[0].tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
        return [
            {"id": i, "text": t, "meta": m, "similarity": 1 - d}
            for i, t, m, d in zip(res["ids"][0], res["documents"][0],
                                  res["metadatas"][0], res["distances"][0])
        ]

    @staticmethod
    def build_context(hits: list[dict]) -> str:
        blocks = []
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            loc = " / ".join(x for x in (m["chapter"], m["section"], m["subsec"]) if x)
            blocks.append(f"[{i}] ({loc})\n{h['text']}")
        return "\n\n".join(blocks)

    def ask(self, query: str, top_k: int = 5, where: dict | None = None,
            show_context: bool = False) -> tuple[str, list[dict]]:
        """(답변, 검색된 청크들) 을 함께 반환 → 화면에서 근거 표시용"""
        hits = self.search(query, top_k, where)
        context = self.build_context(hits)
        if show_context:
            print("[검색된 조항] " + ", ".join(h["meta"]["article"] for h in hits))

        t = time.time()
        res = self.llm.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[근거]\n{context}\n\n[질문]\n{query}"},
            ],
        )
        if show_context:
            print(f"[생성 {time.time() - t:.1f}초]")
        return res.choices[0].message.content, hits


if __name__ == "__main__":
    bot = RagBot()
    print(f"준비 완료 ({bot.collection.count()}개 청크)")
    while True:
        q = input("\n질문 (종료: q) > ").strip()
        if q.lower() == "q":
            break
        if q:
            answer, _ = bot.ask(q, show_context=True)
            print(answer)
