"""
答案生成（零第三方依赖）

两条路径：
1. 抽取式生成（默认）：把检索到的文档切句，按查询词 IDF 加权打分，挑最相关的句子
   组织成答案，并标注来源编号与知识图谱信息 —— 无需任何模型即可产出有信息量的回答
2. 可选 LLM：用标准库 urllib 直连 OpenAI 兼容接口（复用 model_manager 的模型池，
   自动切换与失败冷却），失败时自动回退到抽取式生成
"""

import json
import re
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

from pure_rag.text_utils import Tokenizer, expand_query, split_sentences_with_offsets

# 结构化字段前缀：命中这些标签的句子更可能是"答案句"
KEY_LABELS = ("解决方案", "答案", "处理", "原因", "诊断", "预防", "症状", "步骤", "结论", "建议")

# 过短的句子（如"如何处理？"）语义不足以作为答案要点
MIN_SENTENCE_TOKENS = 3

# 生成答案时忽略的句子前缀（避免把检索到的提问原样回显给用户）
SKIP_PREFIXES = ("问题：", "问题:")

# 疑问词：以问号结尾且含疑问词的句子是"提问"而非"答案"
QUESTION_WORDS = ("什么", "如何", "怎么", "为什么", "哪些", "是否", "多少", "哪些", "怎样")

# 过短的句子不足以作为答案要点
MIN_SENTENCE_CHARS = 8

# 相关性下限：低于最高分该比例的文档不参与答案组织（MMR 引入的多样性文档常在此列）
RELEVANCE_FLOOR = 0.2

# 生成答案时的提示词
SYSTEM_PROMPT = (
    "你是风电领域的运维专家助手。请严格依据给定的参考资料回答问题，"
    "用简洁专业的中文作答；资料中没有的内容不要编造，并说明信息不足。"
)

# 故障案例的结构化字段（由 data_processing 的格式化模板产出，每行一个「字段：值」）
FAULT_LABELS = ("故障ID", "部件", "故障类型", "症状", "原因", "诊断",
                "解决方案", "预防措施", "严重程度")
CARD_LABELS = ("症状", "原因", "诊断", "解决方案", "预防措施")


def parse_fault_cases(content: str) -> List[Dict[str, str]]:
    """
    从文本中解析故障案例（一个分块里可能正好装下多个案例）

    只保留字段数足够多的块，避免把普通文档里恰好出现的「原因：」误判成案例。
    """
    cases, current = [], {}
    for line in content.splitlines():
        for label in FAULT_LABELS:
            prefix = f"{label}："
            if line.startswith(prefix):
                if label == "故障ID" and current:
                    cases.append(current)
                    current = {}
                current[label] = line[len(prefix):].strip()
                break
    if current:
        cases.append(current)
    return [case for case in cases if len(case) >= 4]


def build_fault_card(documents: List[Dict]) -> Optional[Dict]:
    """取排序最靠前的故障案例，整理成结构化卡片（症状/原因/诊断/解决方案/预防措施）"""
    for document in documents:
        cases = parse_fault_cases(document.get("content", ""))
        if not cases:
            continue
        case = cases[0]
        component = case.get("部件", "")
        fault_type = case.get("故障类型", "")
        return {
            "title": " · ".join(part for part in (component, fault_type) if part) or "故障案例",
            "component": component,
            "fault_type": fault_type,
            "severity": case.get("严重程度", ""),
            "doc_id": document.get("doc_id"),
            "fields": [{"label": label, "value": case[label]}
                       for label in CARD_LABELS if case.get(label)],
        }
    return None


# 错误体回读上限（字符），避免异常信息里塞进整个响应
MAX_ERROR_BODY_CHARS = 300

# 思维链模型（如 glm-z1-flash）会把推理过程直接写在正文里返回，
# 送给用户前得切掉，否则答案开头就是一段 <think> 推演
THINK_CLOSE_TAG = "</think>"

# 模型有时在末尾把资料编号单独秃成一行（"[1] [2] [3]"、或带“资料编号：”前缀），
# 对读者没有信息量，去掉
CITATION_TAIL_RE = re.compile(
    r"(?:\n\s*(?:资料编号|引用编号|参考文献|references)?[：:]?\s*(?:\[\d+\][\s,，、]*)+\s*)+$",
    re.IGNORECASE)


class LLMClient:
    """
    OpenAI 兼容接口客户端（urllib 实现）

    模型池来自 src/model_manager.py（本身零依赖），支持失败自动切换模型。
    """

    def __init__(self, model_manager=None, timeout: int = 90, max_attempts: int = 3):
        self.model_manager = model_manager
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        if not self.model_manager or not self.model_manager.models:
            return False
        return self.model_manager.get_current_model() is not None

    def chat(self, prompt: str, system_prompt: str = SYSTEM_PROMPT,
             temperature: float = 0.7, max_tokens: int = 600) -> Optional[str]:
        """调用对话接口，全部模型失败时返回 None（失败原因留在 last_error）"""
        for _ in range(self.max_attempts):
            model = self.model_manager.get_current_model()
            if not model:
                break
            try:
                content = self._post(model, prompt, system_prompt, temperature, max_tokens)
                self.model_manager.report_success(model.name)
                return content
            except Exception as exc:                      # noqa: BLE001 - 任何异常都触发降级
                self.last_error = f"{model.name}: {exc}"
                self.model_manager.report_failure(model.name, str(exc))
        return None

    @staticmethod
    def _describe_http_error(exc: urllib.error.HTTPError) -> str:
        """把服务端错误体里的 code/msg 取出来，区分过载/无权限/余额不足这类不同失败原因"""
        try:
            body = exc.read().decode("utf-8", "ignore")[:MAX_ERROR_BODY_CHARS]
        except OSError:
            body = ""
        detail = body.strip()
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                error = parsed.get("error", parsed)
                if isinstance(error, dict):
                    code = error.get("code") or error.get("err_code") or parsed.get("code") or ""
                    message = error.get("message") or error.get("msg") or ""
                    detail = " ".join(part for part in (str(code), message) if part) or detail
        except ValueError:
            pass
        return f"HTTP {exc.code} {detail}".strip()

    def _post(self, model, prompt: str, system_prompt: str,
              temperature: float, max_tokens: int) -> str:
        payload = json.dumps({
            "model": model.name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        request = urllib.request.Request(
            url=f"{model.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {model.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 带上服务端给出的 code/msg，否则只能看到 "HTTP Error 429" 分不出是过载还是收费档
            # 模型名由 chat() 统一前缀，这里不重复
            raise RuntimeError(self._describe_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"网络不可达（{exc.reason}）") from exc
        except TimeoutError as exc:
            # urllib 在 urlopen 阶段抛 HTTPError/URLError，Body 读满时可能冒 TimeoutError
            raise RuntimeError(f"响应超时（{self.timeout}s）") from exc

        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            # 思维链模型有时只填 reasoning_content，不填 content
            content = (message.get("reasoning_content") or "").strip()
        if THINK_CLOSE_TAG in content:
            content = content.rsplit(THINK_CLOSE_TAG, 1)[1].strip()
        content = CITATION_TAIL_RE.sub("", content).strip()
        if not content:
            raise RuntimeError(f"返回内容为空（finish_reason="
                               f"{data['choices'][0].get('finish_reason')}）")
        return content


class AnswerGenerator:
    """答案生成器：抽取式优先，可选接入 LLM"""

    def __init__(self, tokenizer: Tokenizer = None, max_sentences: int = 6,
                 llm_client: LLMClient = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.max_sentences = max_sentences
        self.llm_client = llm_client

    # -- 对外接口 -----------------------------------------------------------
    def generate(self, question: str, documents: List[Dict], kg_context: str = "",
                 idf_lookup: Callable[[str], float] = None) -> Dict:
        """
        返回：
            answer    答案文本
            mode      llm | extractive | empty
            citations 来源条目（编号/来源/标题/分数）
            claims    抽取式要点，每条带 doc_id 与在原文中的字符区间，可跳转定位
            card      故障结构化卡片（检索到故障案例时才有）
        """
        if not documents and not kg_context:
            return {
                "answer": "知识库中没有检索到相关内容，请尝试换个说法或补充具体部件名称。",
                "mode": "empty",
                "citations": [],
                "claims": [],
                "card": None,
            }

        documents = self._filter_by_score(documents)
        card = build_fault_card(documents)
        llm_error = None
        if self.llm_client:
            if self.llm_client.available:
                answer = self.llm_client.chat(self._build_prompt(question, documents, kg_context))
                if answer:
                    return {"answer": answer, "mode": "llm",
                            "citations": self._citations(documents),
                            "claims": [], "card": card}
                llm_error = self.llm_client.last_error
            else:
                # 模型池全部处于冷却期时连试都不试；不告知的话用户只看得到 extractive，
                # 不知道是大模型挂了还是本来就没配；说清“本轮未调用”，否则与几毫秒的耗时并列会矛盾
                last = self.llm_client.last_error
                llm_error = ("模型池不可用（冷却中），本轮未调用大模型"
                             + (f"；最近一次失败：{last}" if last else ""))

        result = self._extractive(question, documents, kg_context, idf_lookup, card)
        if llm_error:
            result["llm_error"] = llm_error
        return result

    # -- 抽取式生成 ---------------------------------------------------------
    def _extractive(self, question: str, documents: List[Dict], kg_context: str,
                    idf_lookup: Callable[[str], float] = None, card: Dict = None) -> Dict:
        idf_lookup = idf_lookup or (lambda _term: 1.0)
        # 打分词表包含同义词扩展结果，保证"变速箱"能匹配到"齿轮箱"的句子
        query_terms = set(self.tokenizer.cut_for_index(question))
        for variant in expand_query(question):
            query_terms |= set(self.tokenizer.cut_for_index(variant))

        # 每个候选要点记录：句子 + 所属文档 + 在原文中的字符区间
        scored: List[tuple] = []
        for doc_rank, document in enumerate(documents):
            content = document["content"]
            for start, end, sentence in split_sentences_with_offsets(content):
                if self._skip_sentence(sentence):
                    continue
                score = self._score_sentence(sentence, query_terms, idf_lookup, doc_rank)
                if score > 0:
                    scored.append((score, doc_rank, start, end, sentence, document))

        # 按分数选句，并过滤高度重复的句子
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        picked, seen = [], []
        for item in scored:
            tokens = set(self.tokenizer.cut(item[4]))
            if any(self._jaccard(tokens, other) > 0.75 for other in seen):
                continue
            seen.append(tokens)
            picked.append(item)
            if len(picked) >= self.max_sentences:
                break

        if not picked:
            # 兜底：直接给出最相关文档的开头片段
            head = documents[0]["content"][:200].replace("\n", " ")
            body = f"未匹配到直接答案，最相关资料（{self._source_name(documents[0])}）：{head}..."
            return {"answer": body, "mode": "extractive",
                    "citations": self._citations(documents),
                    "claims": [], "card": card}

        # 按文档顺序还原，阅读更连贯
        picked.sort(key=lambda item: (item[1], item[2]))
        claims, lines = [], []
        for index, (_score, _doc_rank, start, end, sentence, document) in enumerate(picked, 1):
            metadata = document.get("metadata", {})
            claims.append({
                "index": index,
                "text": sentence,
                "doc_id": document.get("doc_id"),
                "source": metadata.get("source", "unknown"),
                "title": metadata.get("title", ""),
                "start": start,
                "end": end,
            })
            lines.append(f"{index}. {sentence}  ——（{self._source_name(document)}）")

        parts = []
        if kg_context:
            parts.append(f"【知识图谱】\n{kg_context}")
        parts.append("【参考要点】\n" + "\n".join(lines))
        parts.append("以上要点摘自知识库检索结果，可结合下方来源条目核对原文。")

        return {
            "answer": "\n\n".join(parts),
            "mode": "extractive",
            "citations": self._citations(documents),
            "claims": claims,
            "card": card,
        }

    def _score_sentence(self, sentence: str, query_terms: set,
                        idf_lookup: Callable[[str], float], doc_rank: int) -> float:
        """句子得分 = 查询词 IDF 权重之和 / 长度惩罚 + 结构标签加分 / 排序位惩罚"""
        tokens = self.tokenizer.cut(sentence)
        if not tokens:
            return 0.0

        hit = sum(idf_lookup(token) for token in tokens if token in query_terms)
        if hit <= 0:
            return 0.0

        score = hit / (len(tokens) ** 0.5)
        if sentence.startswith(KEY_LABELS):
            score *= 1.5
        score *= 1.0 / (1.0 + 0.15 * doc_rank)
        return score

    @staticmethod
    def _skip_sentence(sentence: str) -> bool:
        """过滤提问句与过短碎片，避免它们混进答案要点"""
        text = sentence.strip()
        if len(text) < MIN_SENTENCE_CHARS or text.startswith(SKIP_PREFIXES):
            return True
        if text.endswith(("？", "?")) and any(word in text for word in QUESTION_WORDS):
            return True
        return False

    @staticmethod
    def _filter_by_score(documents: List[Dict], ratio: float = RELEVANCE_FLOOR) -> List[Dict]:
        """过滤掉分数远低于最高分的文档（通常是重排为多样性引入的弱相关文档）"""
        if not documents:
            return []
        top = max(float(doc.get("score", 0.0)) for doc in documents)
        if top <= 0:
            return list(documents)
        kept = [doc for doc in documents if float(doc.get("score", 0.0)) >= ratio * top]
        return kept or [documents[0]]

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _source_name(document: Dict) -> str:
        metadata = document.get("metadata", {})
        source = metadata.get("source", "未知来源")
        title = metadata.get("title") or ""
        return f"{source}/{title}" if title and title != source else str(source)

    def _citations(self, documents: List[Dict]) -> List[Dict]:
        return [
            {
                "index": index,
                # doc_id 用于把引用编号回映射到检索列表，Web 才能标注哪篇是 [N]
                "doc_id": document.get("doc_id"),
                "source": document.get("metadata", {}).get("source", "unknown"),
                "title": document.get("metadata", {}).get("title", ""),
                "score": round(float(document.get("score", 0.0)), 4),
            }
            for index, document in enumerate(documents, 1)
        ]

    # -- LLM 生成 -----------------------------------------------------------
    def _build_prompt(self, question: str, documents: List[Dict], kg_context: str) -> str:
        """
        资料编号列表：知识图谱作为最后一条资料入列，而不是另起【知识图谱信息】段落。

        单独开一段带标题的区块时，模型会把“【知识图谱信息】”这类内部标签当正文引用出来。
        """
        blocks = []
        for index, document in enumerate(documents, 1):
            blocks.append(f"[资料{index}] 来源：{self._source_name(document)}\n{document['content']}")
        if kg_context:
            blocks.append(f"[资料{len(blocks) + 1}] 来源：知识图谱\n{kg_context}")
        context = "\n\n".join(blocks)
        return (
            f"参考资料如下：\n\n{context}\n\n"
            f"用户问题：{question}\n\n"
            "请基于上述资料作答，并在关键结论后标注资料编号（如 [1]）；"
            "不要把「资料」「来源」这类标签写进答案。"
        )
