"""검색 + 답변 생성 (멀티턴)
터미널 질의응답:  python rag.py
Streamlit 에서는 RagBot, Memory 를 불러다 씀
"""
import os, re, time
import chromadb
from openai import OpenAI

from common import active_db_dir, STRATEGIES, DOCUMENTS, LENGTH_DOCUMENT, load_embedder, embed_texts

LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """당신은 법률 안내 챗봇입니다. [근거]에는 서로 다른 법률의 조문이 함께 담길 수 있습니다.
- 반드시 아래 [근거] 내용만 근거로 답하세요.
- 이전 대화는 질문의 의도를 파악하는 데만 쓰고, 사실 근거는 이번 [근거]에서만 가져오세요.
- 이전에 무엇을 물었는지 되짚어 설명하지 말고, 조문 내용으로 바로 답하세요.
- 조문의 표현을 최대한 그대로 사용하고, 조문에 없는 사실이나 수치를 새로 만들어내지 마세요.
- 여러 조문을 나란히 비교하거나 정리하는 것은 괜찮습니다. 이때도 각 내용은 해당 조문의 표현을 따르세요.
- 근거에 없는 내용이면 '문서에서 확인되지 않습니다.'라고만 답하세요.
- <개정 ...>, <신설 ...> 같은 개정 이력 표기는 답변에 옮기지 마세요. 단, 개정·신설 시기를 묻는 질문에는 이를 근거로 답하세요.
- 조문이 직접 규정하지 않은 법적 결론(두 제도를 동시에 적용받는지, 무엇이 우선인지 등)은 단정하지 마세요. 관련 조문 내용을 각각 제시한 뒤 '그 관계는 조문에 직접 규정되어 있지 않습니다.'라고 밝히세요.
- 근거 줄에는 [근거]에 실제로 있는 조항과 항만 적으세요. 제125조와 제125조의2처럼 번호가 비슷해도 서로 다른 조문이니 구분하고, 서로 다른 법률에 같은 번호의 조문이 있으면 반드시 법률명으로 구분하세요.
- 답변 본문 다음 줄에, 근거를 반드시 한 줄로 아래 형식 그대로 적으세요. 각 조문 앞에 [근거]에 표시된 법률명을 반드시 붙이고, 조문 제목·괄호·①② 같은 기호는 쓰지 말고 '법률명 제N조 제M항' 형태로만 적으세요.
  근거: 저작권법 제39조 제1항
  근거: 근로기준법 제48조 제1항, 제2항
  근거: 저작권법 제10조, 근로기준법 제16조"""

# 이전 대화를 가리키는 단서: 지시어, 순서 표현, 비교 표현, "공동저작물은요?" 같은 생략형 끝맺음
REF_CUE = re.compile(r"그|이거|저거|아까|방금|처음|먼저|앞에|위에|둘|차이|달라|다른가|비교|관계|예외|또|"
                     r"(?:은|는)(?:요)?\?\s*$")

REWRITE_PROMPT = """당신은 검색어 재작성기입니다. 질문에 답하는 것이 아니라, 검색에 쓸 질문 문장을 만드는 것이 역할입니다.
- [마지막 질문]을 [최근 대화]와 [지금까지 다룬 조항]을 참고해, 이전 대화 없이도 뜻이 통하는 하나의 질문으로 다시 쓰세요.
- '그 조항', '그거', '그럼' 같은 지시어는 특별한 단서가 없으면 가장 최근(직전) 질문의 대상을 가리킵니다.
- 지시어가 하나를 가리키면 대상도 하나만 적으세요. 여러 대상을 섞지 마세요.
- '처음 거'처럼 순서를 말하면 (N번째 질문) 번호를 보고 해당 턴의 대상을 적으세요.
- 비교 질문에서 비교 대상 하나가 생략되어 있으면, 생략된 쪽은 직전 질문의 대상으로 보세요.
- 이전 대화의 대상을 가리킬 때는, 그 턴 답변의 '근거:' 줄에 있는 조문 번호를 괄호로 함께 적으세요.
- 이전 대화를 가리키는 질문이면, 대화 자체(무엇을 물었는지)를 묻는 질문으로 바꾸지 말고 조문 내용을 묻는 질문으로 쓰세요.
- "처음 질문이 뭐였어?"처럼 특정 턴 하나를 콕 집어 다시 묻는 질문이면, 그 턴 하나의 대상만 쓰세요. 다른 턴에서 다룬 주제를 함께 섞지 마세요.
- 법률과 관련 없는 말(잡담, 다른 주제)이면 이전 대화와 억지로 연결하지 말고 그대로 출력하세요.
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
출력: 제30조가 뭐야?

예시 4)
(1번째 질문) 출판권은 누가 설정해?       → 근거: 제63조
마지막 질문: 오늘 날씨 좋네요?
출력: 오늘 날씨 좋네요?

예시 5)
(1번째 질문) 저작권은 사후 얼마나 유지되나요?  → 근거: 제39조 제1항, 제2항
(2번째 질문) 제30조가 뭐야?                     → 근거: 제30조
마지막 질문: 처음 질문이 뭐였는지 다시 말해줘
출력: 저작권의 보호기간(제39조)은 어떻게 되나요?"""

# 질문 속 조문 번호: "제39조", "39조", "제104조의2", "104조 의 2" 모두 허용
ART_NO = re.compile(r"제?\s*(\d+)\s*조(?:\s*의\s*(\d+))?")
DOC_IN_QUERY = re.compile("|".join(re.escape(name) for name in sorted(DOCUMENTS.values(), key=len, reverse=True)))


def article_requests(query: str) -> list[tuple[str | None, str]]:
    docs = list(DOC_IN_QUERY.finditer(query))
    current_doc, doc_index = None, 0
    requests = []
    for m in ART_NO.finditer(query):
        while doc_index < len(docs) and docs[doc_index].end() <= m.start():
            current_doc = docs[doc_index].group()
            doc_index += 1
        number, sub = m.groups()
        requests.append((current_doc, f"제{number}조" + (f"의{sub}" if sub else "")))
    return requests


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
_DOC_NAMES = "|".join(re.escape(name) for name in sorted(DOCUMENTS.values(), key=len, reverse=True))
CITE_TOKEN = re.compile(
    rf"(?:({_DOC_NAMES})\s*)?(제\d+조(?:의\d+)?)?\s*(?:제(\d+)항)?\s*(?:제(?:(\d+(?:의\d+)?)호|(\d+)호의(\d+)))?"
)
HO_NO = re.compile(r"^\s*(\d+(?:의\d+)?)\.\s", re.M)


def parse_citations(answer: str) -> list[tuple[str | None, str, str | None, str | None]]:
    """답변의 모든 '근거:' 줄을 읽어 [(법률명 또는 None, 조문번호, 항번호 또는 None, 호번호 또는 None), ...] 로 반환.
    형식이 흔들려도(여러 줄, 제목 괄호, ①② 기호, 법률명·항 생략) 최대한 읽어낸다."""
    lines = [l.split(":", 1)[1] for l in answer.splitlines() if l.strip().startswith("근거:")]
    out = []
    for line in lines:
        line = re.sub(r"\([^)]*\)", "", line).replace("*", "")
        for i, ch in enumerate(CIRCLED, 1):
            line = line.replace(ch, f" 제{i}항")
        last_doc, last_art, last_para = None, None, None
        for token in re.split(rf"[,·]|\s+및\s+(?=(?:{_DOC_NAMES})|제\d+(?:조|항|호))", line):
            token = token.strip()
            if not token:
                continue
            m = CITE_TOKEN.fullmatch(token)
            if not m:
                unknown = re.fullmatch(r"(.+?)\s+(제\d+조(?:의\d+)?)(?:\s*제(\d+)항)?(?:\s*제(?:(\d+(?:의\d+)?)호|(\d+)호의(\d+)))?", token)
                if unknown:
                    ho = unknown.group(4) or (f"{unknown.group(5)}의{unknown.group(6)}" if unknown.group(5) else None)
                    out.append((*unknown.group(1, 2, 3), ho))
                else:
                    out.append((None, f"인용 형식 미확인: {token}", None, None))
                continue
            doc = m.group(1) or last_doc                          # 법률명 생략 시 직전 것 재사용
            art = m.group(2) or last_art                          # "제2항"만 있으면 앞 조문에 붙임
            if not art:
                out.append((None, f"인용 형식 미확인: {token}", None, None))
                continue
            para = m.group(3) or (last_para if not m.group(2) else None)  # "제8호"만 있으면 앞 항에 붙임
            last_doc, last_art, last_para = doc, art, para
            ho = m.group(4) or (f"{m.group(5)}의{m.group(6)}" if m.group(5) else None)
            out.append((doc, art, para, ho))
    return out


def cited_articles(answer: str, hits: list[dict]) -> list[str]:
    """인용된 법률·조문을 검색된 청크의 정확한 표시명으로 변환 (기억 저장용)"""
    titles = []
    for doc, art, _, _ in parse_citations(answer):
        for h in hits:
            m = h["meta"]
            if art in m.get("article_nos", [m["article_no"]]) and (doc is None or m["doc"] == doc):
                title = f"{m['doc']} {m['article'] if m['article_no'] == art else art}"
                if title not in titles:
                    titles.append(title)
                break
    return titles


class RagBot:
    """무거운 자원(LLM 클라이언트, 임베딩 모델, DB)을 한 번 로딩해서 들고 있는 객체"""

    def __init__(self, strategy: str = "article"):
        if strategy not in STRATEGIES:
            raise ValueError(f"알 수 없는 청킹 전략: {strategy} (선택지: {list(STRATEGIES)})")
        self.strategy = strategy

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(".env 에서 OPENAI_API_KEY 를 못 읽었습니다")
        self.llm = OpenAI(api_key=key)
        self.embedder = load_embedder()

        client = chromadb.PersistentClient(path=active_db_dir())
        try:
            self.collection = client.get_collection(STRATEGIES[strategy])   # 열기만, 지우지 않음
            self.citation_collection = client.get_collection(STRATEGIES["article"])
        except Exception as e:
            raise RuntimeError("컬렉션이 없습니다. 먼저 python build_index.py 를 실행하세요") from e

    def lookup(self, query: str, where: dict | None = None) -> list[dict]:
        """질문에 조문 번호가 있으면 메타데이터 article_no 로 그 조문의 청크를 직접 가져옴"""
        found = []
        for doc, key in article_requests(query):
            cond = {"$or": [{"article_no": key}, {"article_nos": {"$contains": key}}]}
            if doc:
                cond = {"$and": [cond, {"doc": doc}]}
            if where:
                cond = {"$and": [cond, where]}               # 부칙 제외 등 기존 필터와 함께
            got = self.collection.get(where=cond, include=["documents", "metadatas"])
            rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                          key=lambda r: int(r[0].split("_")[1]))   # ①②③ 순서 유지
            found += [{"id": i, "text": t, "meta": m, "similarity": None} for i, t, m in rows]
        return list({h["id"]: h for h in found}.values())

    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
        """번호 직접 조회 결과를 앞에, 나머지를 벡터 검색으로 채움 (유사도 = 1 - distance)"""
        if getattr(self, "strategy", "article") == "length":
            if any(doc != DOCUMENTS[LENGTH_DOCUMENT] for doc in DOC_IN_QUERY.findall(query)):
                return []
        filters = where.get("$and", [where]) if where else []
        allowed = next((part["doc"]["$in"] for part in filters
                        if "doc" in part and isinstance(part["doc"], dict) and "$in" in part["doc"]), None)
        if allowed is not None and any(doc not in allowed for doc, _ in article_requests(query) if doc):
            return []
        direct = self.lookup(query, where)
        if direct and all(doc is None for doc, _ in article_requests(query)):
            direct = direct[:top_k]
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

    def verify_citations(self, answer: str, hits: list[dict]) -> list[str]:
        """인용한 법률·조·항·호가 실제로 존재하는지 대조. hits는 "이 조문이 근거로 쓰였나"
        (환각 방지)만 확인하는 데 쓰고, 항·호 번호가 몇 개까지 있는지는 top_k 검색 결과에
        갇히지 않게 DB에서 해당 조문 전체를 직접 조회해 확인한다 — 조문이 여러 청크로
        나뉘어 있을 때 그중 일부만 검색에 뽑혀서 생기는 오탐을 막는다.
        호는 그 조문 전체 기준으로 존재 여부만 확인한다 (어느 항 소속인지까지는 안 따짐)."""
        hit_keys = {(m["doc"], art, m.get("supplement", ""))
                    for h in hits for m in [h["meta"]]
                    for art in m.get("article_nos", [m["article_no"]]) if art}
        detail_cache: dict[tuple[str, str, str], tuple[set, set]] = {}   # key -> (항 번호 집합, 호 번호 집합)

        bad = []
        for doc, art, para, ho in parse_citations(answer):
            label = f"{doc + ' ' if doc else ''}{art}"
            if para:
                label += f" 제{para}항"
            if ho:
                label += f" 제{ho}호"
            candidates = [k for k in hit_keys if k[1] == art and (doc is None or k[0] == doc)]
            if len(candidates) > 1:
                bad.append(label + " (근거 범위 불명확)")
                continue
            if not candidates:
                bad.append(label)
                continue
            if not para and not ho:
                continue

            key = candidates[0]
            if key not in detail_cache:
                cond = {"$and": [{"doc": key[0]}, {"article_no": key[1]},
                                 {"supplement": key[2]}]}
                got = getattr(self, "citation_collection", self.collection).get(where=cond, include=["documents"])
                paras, hos = set(), set()
                for text in got["documents"]:
                    paras.update(CIRCLED.index(ch) + 1 for ch in text if ch in CIRCLED)
                    hos.update(HO_NO.findall(text))
                detail_cache[key] = (paras, hos)
            paras, hos = detail_cache[key]
            if (para and int(para) not in paras) or (ho and ho not in hos):
                bad.append(label)
        return list(dict.fromkeys(bad))                               # 중복 제거, 순서 유지

    @staticmethod
    def build_context(hits: list[dict]) -> str:
        blocks = []
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            loc = " / ".join(x for x in (m["doc"], m["chapter"], m["section"], m["subsec"]) if x)
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
        # 안전장치 1: 질문 형태가 아니면(재작성기가 답을 써버린 경우 등) 원래 질문으로
        if not out or not out.rstrip().endswith("?"):
            return query
        # 안전장치 2: 원래 질문에 이전 대화 단서가 없는데 조문 번호가 새로 생겼으면 억지 연결로 보고 원래 질문으로
        added = set(ART_NO.findall(out)) - set(ART_NO.findall(query))
        if added and not REF_CUE.search(query):
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
        unverified = self.verify_citations(answer, hits)
        if show_context:
            print(f"[생성 {time.time() - t:.1f}초] 인용: {', '.join(cited) or '없음'}")
            if unverified:
                print(f"[경고] 검색된 근거에 없는 인용: {', '.join(unverified)}")

        if memory is not None:
            memory.add(query, answer, cited)

        return {"answer": answer, "hits": hits, "search_query": search_query,
                "cited": cited, "unverified": unverified}


if __name__ == "__main__":
    s = input(f"청킹 전략 선택 {list(STRATEGIES)} (엔터=article) > ").strip() or "article"
    bot = RagBot(strategy=s)
    memory = Memory()
    print(f"[{bot.strategy}] 준비 완료 ({bot.collection.count()}개 청크)")
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
