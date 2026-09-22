"""검색 + 답변 생성 (멀티턴)
터미널 질의응답:  python rag.py
Streamlit 에서는 RagBot, Memory 를 불러다 씀
"""
import os, re, time
import chromadb
from openai import OpenAI

from common import DB_DIR, COLLECTION, load_embedder, embed_texts

LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """당신은 저작권법 안내 챗봇입니다.
- 반드시 아래 [근거] 내용만 근거로 답하세요.
- 이전 대화는 질문의 의도를 파악하는 데만 쓰고, 사실 근거는 이번 [근거]에서만 가져오세요.
- 이전에 무엇을 물었는지 되짚어 설명하지 말고, 조문 내용으로 바로 답하세요.
- 조문의 표현을 최대한 그대로 사용하고, 조문에 없는 사실이나 수치를 새로 만들어내지 마세요.
- 여러 조문을 나란히 비교하거나 정리하는 것은 괜찮습니다. 이때도 각 내용은 해당 조문의 표현을 따르세요.
- 근거에 없는 내용이면 '문서에서 확인되지 않습니다.'라고만 답하세요.
- <개정 ...>, <신설 ...> 같은 개정 이력 표기는 답변에 옮기지 마세요. 단, 개정·신설 시기를 묻는 질문에는 이를 근거로 답하세요.
- 근거 줄에는 [근거]에 실제로 있는 조항과 항만 적으세요. 조문 번호는 [근거]의 조문 제목에 적힌 그대로 쓰세요(예: 제125조와 제125조의2는 서로 다른 조문입니다).
- 답변 본문 다음 줄에, 아래 형식 그대로 근거를 한 줄로 적으세요. 괄호나 다른 문구를 붙이지 마세요.
  근거: 제39조 제1항
  근거: 제48조 제1항, 제2항
  근거: 제10조, 제16조 제1항"""

REWRITE_PROMPT = """당신은 검색어 재작성기입니다. 질문에 답하는 것이 아니라, 검색에 쓸 질문 문장을 만드는 것이 역할입니다.
- [마지막 질문]을 [최근 대화]와 [지금까지 다룬 조항]을 참고해, 이전 대화 없이도 뜻이 통하는 하나의 질문으로 다시 쓰세요.
- '그 조항', '그거', '그럼' 같은 지시어는 특별한 단서가 없으면 가장 최근(직전) 질문의 대상을 가리킵니다.
- 지시어가 하나를 가리키면 대상도 하나만 적으세요. 여러 대상을 섞지 마세요.
- '처음 거'처럼 순서를 말하면 (N번째 질문) 번호를 보고 해당 턴의 대상을 적으세요.
- 비교 질문에서 비교 대상 하나가 생략되어 있으면, 생략된 쪽은 직전 질문의 대상으로 보세요.
- 이전 대화의 대상을 가리킬 때는, 그 턴 답변의 '근거:' 줄에 있는 조문 번호를 괄호로 함께 적으세요.
- 대화 자체(무엇을 물었는지)를 묻는 질문으로 바꾸지 말고, 조문 내용을 묻는 질문으로 쓰세요.
- 마지막 질문에 지시어가 없고 이미 혼자서 뜻이 통하면, 한 글자도 바꾸지 말고 그대로 출력하세요.
- 답하지 말고, 물음표(?)로 끝나는 질문 한 줄만 출력하세요.

예시 1)
(1번째 질문) 도서관에서 복제할 수 있는 경우는?  → 근거: 제31조
(2번째 질문) 학교 수업에서는?                    → 근거: 제25조
마지막 질문: 처음 거랑 뭐가 달라?
출력: 도서관 등에서의 복제(제31조)와 학교교육 목적 이용(제25조)의 허용 범위는 어떻게 다른가요?

예시 2)
(1번째 질문) 저작인격권이 뭐야?          → 근거: 제11조, 제12조, 제13조
(2번째 질문) 제23조가 뭐야?              → 근거: 제23조
마지막 질문: 그거 예외는 없어?
출력: 재판 등에서의 복제(제23조)에 예외가 있나요?

예시 3)
(1번째 질문) 영상저작물이 뭐야?          → 근거: 제2조
마지막 질문: 제30조가 뭐야?
출력: 제30조가 뭐야?"""

CITE = re.compile(r"제\d+조(?:의\d+)?")
# 질문 속 조문 번호: "제39조", "39조", "제104조의2", "104조 의 2" 모두 허용
ART_NO = re.compile(r"제?\s*(\d+)\s*조(?:\s*의\s*(\d+))?")


class Memory:
    """대화 기억: 최근 대화 원문 + 지금까지 다룬 조항 목록"""

    def __init__(self, max_turns: int = 3, max_topics: int = 8):
        self.max_turns = max_turns
        self.max_topics = max_topics
        self.count = 0                           # 지금까지 총 몇 턴 했는지
        self.turns: list[tuple[str, str]] = []   # (질문, 답변)
        self.topics: list[str] = []              # "제39조(보호기간의 원칙)" 형태, 오래된 것 → 최근 것

    def add(self, question: str, answer: str, cited: list[str]):
        self.count += 1
        self.turns.append((question, answer))
        self.turns = self.turns[-self.max_turns:]          # 최근 N턴만 유지
        for t in cited:
            if t in self.topics:
                self.topics.remove(t)                       # 다시 나오면 맨 뒤(최근)로
            self.topics.append(t)
        self.topics = self.topics[-self.max_topics:]

    def clear(self):
        self.count, self.turns, self.topics = 0, [], []


CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CITE_TOKEN = re.compile(r"(제\d+조(?:의\d+)?)?\s*(?:제(\d+)항)?")


def verify_citations(answer: str, hits: list[dict]) -> list[str]:
    """'근거:' 줄의 조·항이 검색된 청크에 실제로 있는지 대조. 없는 인용 목록을 반환."""
    line = next((l for l in reversed(answer.splitlines())
                 if l.strip().startswith("근거:")), "")
    if not line:
        return []

    # 검색된 청크에 있는 조문 → 항 번호 집합 (항 기호가 없는 청크는 조문 전체로 간주: None)
    have: dict[str, set] = {}
    for h in hits:
        art = h["meta"]["article"].split("(")[0]
        paras = {CIRCLED.index(ch) + 1 for ch in h["text"] if ch in CIRCLED}
        have.setdefault(art, set()).update(paras or {None})

    bad, last_art = [], None
    for token in line.split(":", 1)[1].split(","):
        m = CITE_TOKEN.search(token.strip())
        art, para = (m.group(1), m.group(2)) if m else (None, None)
        art = art or last_art                      # "제2항"처럼 조문이 생략되면 앞 조문으로
        if not art:
            continue
        last_art = art
        label = f"{art} 제{para}항" if para else art
        if art not in have:
            bad.append(label)
        elif para and int(para) not in have[art] and None not in have[art]:
            bad.append(label)
    return bad


def cited_articles(answer: str, hits: list[dict]) -> list[str]:
    """답변의 '근거:' 줄에서 조문 번호를 뽑아, 검색된 청크의 정확한 조문 제목으로 변환"""
    line = next((l for l in reversed(answer.splitlines())
                 if l.strip().startswith("근거:")), "")
    titles = []
    for num in CITE.findall(line):
        for h in hits:
            title = h["meta"]["article"]
            if title.startswith(num + "(") and title not in titles:
                titles.append(title)
                break
    return titles


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

    def lookup(self, query: str, where: dict | None = None) -> list[dict]:
        """질문에 조문 번호가 있으면 메타데이터 article_no 로 그 조문의 청크를 직접 가져옴"""
        found = []
        for num, sub in ART_NO.findall(query):
            key = f"제{num}조" + (f"의{sub}" if sub else "")
            cond = {"article_no": key}
            if where:
                cond = {"$and": [cond, where]}               # 부칙 제외 등 기존 필터와 함께
            got = self.collection.get(where=cond, include=["documents", "metadatas"])
            rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                          key=lambda r: int(r[0].split("_")[1]))   # ①②③ 순서 유지
            found += [{"id": i, "text": t, "meta": m, "similarity": None} for i, t, m in rows]
        return found

    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
        """번호 직접 조회 결과를 앞에, 나머지를 벡터 검색으로 채움 (유사도 = 1 - distance)"""
        direct = self.lookup(query, where)
        q_emb = embed_texts(self.embedder, [query])[0].tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
        vector = [
            {"id": i, "text": t, "meta": m, "similarity": 1 - d}
            for i, t, m, d in zip(res["ids"][0], res["documents"][0],
                                  res["metadatas"][0], res["distances"][0])
        ]
        seen = {h["id"] for h in direct}
        merged = direct + [h for h in vector if h["id"] not in seen]
        return merged[:max(top_k, len(direct))]   # 직접 조회분은 잘리지 않게

    @staticmethod
    def build_context(hits: list[dict]) -> str:
        blocks = []
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            loc = " / ".join(x for x in (m["chapter"], m["section"], m["subsec"]) if x)
            blocks.append(f"[{i}] ({loc})\n{h['text']}")
        return "\n\n".join(blocks)

    def rewrite(self, query: str, memory: Memory | None) -> str:
        """이전 대화를 반영해 검색용 독립 질문으로 재작성 (첫 질문이면 그대로)"""
        if memory is None or not memory.turns:
            return query
        start = memory.count - len(memory.turns) + 1   # 창이 밀려도 실제 턴 번호 유지
        recent = "\n".join(f"({i}번째 질문) 사용자: {q}\n챗봇: {a}"
                           for i, (q, a) in enumerate(memory.turns, start))
        topics = ", ".join(memory.topics) or "없음"
        res = self.llm.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content":
                    f"[지금까지 다룬 조항]\n{topics}\n\n[최근 대화]\n{recent}\n\n[마지막 질문]\n{query}"},
            ],
        )
        out = res.choices[0].message.content.strip()
        # 안전장치: 질문 형태가 아니면(재작성기가 답을 써버린 경우 등) 원래 질문으로 검색
        if not out or not out.rstrip().endswith("?"):
            return query
        return out

    def ask(self, query: str, memory: Memory | None = None, top_k: int = 5,
            where: dict | None = None, show_context: bool = False) -> dict:
        """재작성 → 검색 → 답변. memory 를 주면 사용 후 이번 턴을 기록한다."""
        search_query = self.rewrite(query, memory)
        hits = self.search(search_query, top_k, where)
        context = self.build_context(hits)
        if show_context:
            if search_query != query:
                print(f"[재작성] {search_query}")
            print("[검색된 조항] " + ", ".join(h["meta"]["article"] for h in hits))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memory:
            for q, a in memory.turns:                   # 지난 턴은 질문·답변만 (근거 제외)
                messages += [{"role": "user", "content": q},
                             {"role": "assistant", "content": a}]
        # 답변에도 재작성된 질문을 사용: 지시어가 이미 풀려 있어 대화를 되짚을 필요가 없음
        messages.append({"role": "user", "content": f"[근거]\n{context}\n\n[질문]\n{search_query}"})

        t = time.time()
        res = self.llm.chat.completions.create(
            model=LLM_MODEL, temperature=0, max_tokens=400, messages=messages,
        )
        answer = res.choices[0].message.content
        cited = cited_articles(answer, hits)
        unverified = verify_citations(answer, hits)
        if show_context:
            print(f"[생성 {time.time() - t:.1f}초] 인용: {', '.join(cited) or '없음'}")
            if unverified:
                print(f"[경고] 검색된 근거에 없는 인용: {', '.join(unverified)}")

        if memory is not None:
            memory.add(query, answer, cited)

        return {"answer": answer, "hits": hits, "search_query": search_query,
                "cited": cited, "unverified": unverified}


if __name__ == "__main__":
    bot = RagBot()
    memory = Memory()
    print(f"준비 완료 ({bot.collection.count()}개 청크)")
    while True:
        q = input("\n질문 (종료: q, 대화 초기화: r) > ").strip()
        if q.lower() == "q":
            break
        if q.lower() == "r":
            memory.clear()
            print("대화를 초기화했습니다.")
            continue
        if q:
            result = bot.ask(q, memory, show_context=True)
            print(result["answer"])
            print(f"[다룬 조항] {', '.join(memory.topics)}")
