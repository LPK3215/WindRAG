"""
纯 Python 文本处理工具（仅标准库）

- 文本归一化：全角转半角、英文小写、空白折叠
- 中文分词：领域词典 + 正向最大匹配 + 未登录词的二元组补充
- 无监督新词发现：词频 + 互信息 + 左右邻接熵（用于自动扩充领域词典）
- 句子切分、查询同义词扩展
"""

import math
import re
from typing import Dict, Iterable, List, Set, Tuple

# ---------------------------------------------------------------------------
# 风电领域词典（部件 / 故障 / 运维 / 性能 / 规范）
# ---------------------------------------------------------------------------
DOMAIN_WORDS: List[str] = [
    # 部件与结构
    "齿轮箱", "增速箱", "主轴", "轴承", "主轴承", "发电机", "叶片", "轮毂", "导流罩",
    "机舱", "塔筒", "基础环", "偏航系统", "偏航轴承", "偏航电机", "变桨系统",
    "变桨轴承", "液压系统", "液压站", "制动系统", "刹车盘", "变压器", "箱变",
    "变频器", "变流器", "控制系统", "主控", "传感器", "风速仪", "风向标", "编码器",
    "联轴器", "弹性支撑", "散热器", "冷却器", "油冷系统", "水冷系统", "过滤器", "滤芯",
    # 故障与状态
    "振动", "异响", "过热", "温度过高", "磨损", "点蚀", "裂纹", "腐蚀", "疲劳",
    "泄漏", "渗漏", "卡涩", "卡滞", "失速", "过载", "短路", "断路", "报警", "停机",
    "绝缘", "绝缘电阻", "油质", "油位", "油温", "铁屑", "齿面", "不对中", "偏载",
    "缺相", "三相不平衡", "失压", "过压", "过流", "欠压", "抱闸", "溜车",
    # 运维与操作
    "润滑", "润滑油", "齿轮油", "润滑脂", "加注", "更换", "检修", "巡检", "保养",
    "大修", "预防性维护", "纠正性维护", "力矩", "螺栓", "紧固", "扭矩", "对中",
    "吊装", "试运行", "空载", "满发", "限功率", "复位", "手动", "自动", "远程",
    # 性能与指标
    "功率", "功率曲线", "发电量", "可利用率", "效率", "风速", "切入风速", "额定风速",
    "切出风速", "转速", "额定转速", "负荷", "载荷", "疲劳寿命", "MTBF", "MTTR",
    # 规范与文档
    "标准", "规程", "手册", "规范", "国标", "行标", "周期", "阈值", "等级",
]

# 停用词（分词后过滤，避免高频虚词干扰 BM25/TF-IDF 打分）
STOPWORDS: Set[str] = {
    # 多字虚词与疑问词
    "什么", "怎么", "如何", "哪些", "是否", "多少", "怎样", "可以", "一个", "进行",
    "相关", "情况", "问题", "时候", "目前", "如果", "因为", "所以", "一般", "通常",
    "应该", "需要", "以及", "通过", "对于", "根据", "由于", "并且", "为什么", "干啥",
    "处理", "方法", "方式", "步骤", "原因", "作用", "功能", "情况", "哪些",
    # 单字虚词、疑问字与通用动词：过滤后"怎么处理""是什么"不再干扰检索
    "怎", "么", "什", "哪", "些", "呀", "呢", "吗", "吧", "啊", "哦", "嗯",
    "处", "理", "办", "法", "方", "式", "做", "说", "看", "想",
    "需", "要", "应", "该", "可", "能", "会", "让", "使", "用", "是", "有", "无",
    "不", "没", "了", "的", "地", "得", "着", "过", "和", "与", "或", "及", "跟",
    "在", "对", "为", "从", "到", "把", "被", "给", "向", "往", "按", "依", "用",
    "就", "都", "也", "还", "很", "太", "更", "最", "才", "再", "又", "而", "但",
    "并", "等", "这", "那", "我", "你", "他", "它", "们", "个", "之", "以", "其", "此",
    "中", "上", "下", "里", "内", "外", "前", "后", "时", "常", "即", "则", "若",
}

# 查询同义词表：口语/别名 -> 领域标准词
# 两个方向都会用到：查询扩展（别名 → 标准词）+ 索引期扩展（标准词 → 别名）
# 只收录真正等义的词，避免"报警↔停机"这类语义不同却被强行打通
SYNONYMS: Dict[str, List[str]] = {
    # 部件别名
    "变速箱": ["齿轮箱"], "减速箱": ["齿轮箱"], "齿轮减速箱": ["齿轮箱"],
    "桨叶": ["叶片"], "风轮": ["叶片", "轮毂"], "叶轮": ["叶片"],
    "偏航": ["偏航系统"], "变桨": ["变桨系统"], "刹车": ["制动系统"],
    "液压": ["液压系统"], "风扇": ["散热器"], "冷却": ["冷却器", "散热器"],
    # 故障别名
    "温度高": ["过热"], "发热": ["过热"], "过热": ["温度过高"],
    "震动": ["振动"], "抖动": ["振动"], "噪音": ["异响"], "噪声": ["异响"],
    "漏油": ["泄漏"], "渗油": ["泄漏"], "漏液": ["泄漏"],
    # 设备别名
    "风力发电机": ["风电机组"], "风机": ["风电机组"], "发电机组": ["发电机"],
    # 操作别名
    "换油": ["更换", "齿轮油"], "加油": ["加注"], "维修": ["检修"], "检查": ["巡检"],
}

# 标点与符号：分词时直接跳过（保留 %、℃ 等与数值连用的单位符号）
PUNCTUATION: Set[str] = set(
    "，。！？；：、,.!?;:\"'`~()（）[]【】{}《》<>-—_/\\|*#@+=^&$…·\t\r\n"
)


def _build_alias_map() -> Dict[str, List[str]]:
    """由 SYNONYMS 反查：标准词 -> 它的别名列表（用于索引期同义词扩展）"""
    mapping: Dict[str, List[str]] = {}
    for alias, standards in SYNONYMS.items():
        for standard in standards:
            mapping.setdefault(standard, [])
            if alias not in mapping[standard]:
                mapping[standard].append(alias)
    return mapping


# 标准词 -> 别名列表；建索引时把别名一并写入，使两种说法可以互相检索到
ALIASES: Dict[str, List[str]] = _build_alias_map()


# 句末标点：切句时归入前一句；换行只作分隔符
_SENTENCE_TERMINATORS = frozenset("。！？；!?;")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-_/%]*")


def normalize(text: str) -> str:
    """全角转半角、英文小写、连续空白折叠"""
    if not text:
        return ""
    chars = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:      # 全角 ASCII
            chars.append(chr(code - 0xFEE0))
        elif code == 0x3000:               # 全角空格
            chars.append(" ")
        else:
            chars.append(ch)
    return re.sub(r"[ \t\u00a0]+", " ", "".join(chars)).lower()


def is_cjk(ch: str) -> bool:
    """判断是否为中日韩统一表意文字"""
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"


class Tokenizer:
    """
    中文分词器：领域词典驱动的前向最大匹配 + 二元组补充

    说明：索引与查询必须使用同一个分词器（同一份词典），否则 BM25/TF-IDF 打分不成立。
    因此新词发现得到的词典需要随索引一起持久化，加载索引时一并恢复。
    """

    def __init__(self, extra_words: Iterable[str] = None, use_bigram: bool = True):
        # 词典 = 领域词 + 同义词表两端（保证"变速箱""漏油"这类说法能整体成词）+ 语料新词
        self.words: Set[str] = set(DOMAIN_WORDS) | set(SYNONYMS) | set(ALIASES)
        if extra_words:
            self.words |= {w for w in extra_words if len(w) >= 2}
        self.use_bigram = use_bigram
        self.max_len = max((len(w) for w in self.words), default=1)
        # 首字 -> 以该字开头的最长词长（匹配时剪枝，避免无谓的切片比较）
        self._first_len: Dict[str, int] = {}
        for word in self.words:
            first = word[0]
            self._first_len[first] = max(self._first_len.get(first, 0), len(word))

    # -- 分词 ---------------------------------------------------------------
    def cut(self, text: str) -> List[str]:
        """基础分词：返回去停用词后的词序列"""
        tokens: List[str] = []
        text = normalize(text)
        i, n = 0, len(text)

        while i < n:
            ch = text[i]

            if ch.isspace() or ch in PUNCTUATION:
                i += 1
                continue

            # 英文/数字/单位整体保留（如 iso、7mm/s、320、85）
            if ch.isascii() and ch.isalnum():
                match = _ASCII_TOKEN_RE.match(text, i)
                end = match.end() if match else i + 1
                token = text[i:end].strip("./%-_")
                if token:
                    tokens.append(token)
                i = end
                continue

            # 剩余符号（℃、%等）当作单字处理
            limit = min(self._first_len.get(ch, 1), self.max_len, n - i)
            matched = ch
            for length in range(limit, 1, -1):
                fragment = text[i:i + length]
                if fragment in self.words:
                    matched = fragment
                    break
            tokens.append(matched)
            i += len(matched)

        return [t for t in tokens if t and t not in STOPWORDS]

    def cut_for_index(self, text: str) -> List[str]:
        """
        索引/查询统一入口，包含三层增强：
        1. 基础分词（领域词典 + 最大匹配）
        2. 相邻单字二元组：缓解"变速箱"被切成单字后的匹配丢失
        3. 同义词扩展：命中标准词时补上它的别名（如 齿轮箱 → 变速箱/减速箱），
           使"变速箱"与"齿轮箱"在 BM25/TF-IDF 中可以直接互相命中
        """
        tokens = self.cut(text)
        expanded = list(tokens)

        if self.use_bigram:
            expanded += [
                a + b
                for a, b in zip(tokens, tokens[1:])
                if len(a) == 1 and len(b) == 1 and is_cjk(a) and is_cjk(b)
            ]

        for token in tokens:
            aliases = ALIASES.get(token)
            if aliases:
                expanded += aliases

        return expanded


def split_paragraphs(text: str) -> List[str]:
    """按行切段落，去掉空行"""
    return [line.strip() for line in text.splitlines() if line.strip()]


def split_sentences_with_offsets(text: str) -> List[Tuple[int, int, str]]:
    """
    按中英文句末标点与换行切句，并保留每句在原文中的字符区间

    返回 [(start, end, sentence)]，start/end 可直接用于高亮定位。
    句末标点归入前一句；换行只作分隔符不计入。
    """
    spans: List[Tuple[int, int, str]] = []
    start = 0

    def _emit(begin: int, end: int) -> None:
        piece = text[begin:end]
        stripped = piece.strip()
        if stripped:
            offset = len(piece) - len(piece.lstrip())
            spans.append((begin + offset, begin + offset + len(stripped), stripped))

    for index, char in enumerate(text):
        if char in _SENTENCE_TERMINATORS:
            _emit(start, index + 1)
            start = index + 1
        elif char == "\n":
            _emit(start, index)
            start = index + 1
    _emit(start, len(text))

    return spans


def split_sentences(text: str) -> List[str]:
    """按中英文句末标点与换行切句，过滤空串"""
    return [sentence for _start, _end, sentence in split_sentences_with_offsets(text)]


def fold_char(ch: str) -> str:
    """单字符归一化：全角转半角 + 小写，且严格保持字符数不变（便于下标对齐）"""
    code = ord(ch)
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    if code == 0x3000:
        return " "
    lowered = ch.lower()
    return lowered if len(lowered) == 1 else ch


def fold_text(text: str) -> str:
    """与 normalize 同效但逐字符处理，保证折半后的下标可映射回原文"""
    return "".join(fold_char(ch) for ch in text)


def find_spans(text: str, terms: Iterable[str]) -> List[Tuple[int, int, str]]:
    """
    在原文中定位查询词的出现位置（返回字符级区间，用于高亮）

    做法：先对原文做逐字符归一化（全角转半角 + 小写，长度不变），
    再用 str.find 做 C 速度的精确匹配，因此下标可以直接映射回原文。
    返回 [(start, end, term)]，重叠区间已合并。
    """
    if not text or not terms:
        return []

    folded = fold_text(text)
    spans: List[Tuple[int, int, str]] = []
    for term in {t for t in terms if t}:
        length = len(term)
        if length == 0 or length > len(text):
            continue
        start = folded.find(term)
        while start != -1:
            spans.append((start, start + length, term))
            start = folded.find(term, start + 1)

    spans.sort(key=lambda span: (span[0], -span[1]))
    merged: List[Tuple[int, int, str]] = []
    for start, end, term in spans:
        if merged and start <= merged[-1][1]:
            # 区间重叠时合并，并保留所有命中的词（"齿轮箱温度过高" → 齿轮箱、温度过高）
            prev_start, prev_end, prev_term = merged[-1]
            joined = prev_term if term in prev_term else f"{prev_term}、{term}"
            merged[-1] = (prev_start, max(end, prev_end), joined)
            continue
        merged.append((start, end, term))
    return merged


def expand_query(query: str) -> List[str]:
    """基于同义词表生成查询扩展（最多 2 条变体，避免噪声过大）"""
    variants: List[str] = []
    for alias, standards in SYNONYMS.items():
        if alias in query:
            for standard in standards:
                replaced = query.replace(alias, standard)
                if replaced != query and replaced not in variants:
                    variants.append(replaced)
    return variants[:2]


# ---------------------------------------------------------------------------
# 无监督新词发现
# ---------------------------------------------------------------------------
def _ngram_counts(runs: List[str], n: int) -> Dict[str, int]:
    """统计所有长度为 n 的汉字串频次"""
    counts: Dict[str, int] = {}
    for run in runs:
        for i in range(len(run) - n + 1):
            gram = run[i:i + n]
            counts[gram] = counts.get(gram, 0) + 1
    return counts


def _entropy(counter: Dict[str, int]) -> float:
    """左右邻接熵：熵越大说明该候选词边界越稳定"""
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def discover_terms(texts: List[str], min_freq: int = 3, min_pmi: float = 3.0,
                   min_entropy: float = 0.6, max_len: int = 4, limit: int = 300) -> List[str]:
    """
    无监督新词发现（频率 + 互信息 + 左右熵）

    只做统计筛选，得到的是"在语料中稳定成词"的候选，用于补充领域词典。
    返回按频次降序排列的候选词列表。
    """
    runs: List[str] = []
    for text in texts:
        runs.extend(_CJK_RUN_RE.findall(normalize(text)))
    if not runs:
        return []

    total_chars = sum(len(run) for run in runs)
    counts_by_len = {n: _ngram_counts(runs, n) for n in range(1, max_len + 1)}
    candidates: List[tuple] = []

    for n in range(2, max_len + 1):
        counts = counts_by_len[n]
        if not counts:
            continue
        # 只对高频候选统计左右邻接字，控制开销
        freq_filtered = {g for g, c in counts.items() if c >= min_freq}
        if not freq_filtered:
            continue

        left: Dict[str, Dict[str, int]] = {}
        right: Dict[str, Dict[str, int]] = {}
        for run in runs:
            for i in range(len(run) - n + 1):
                gram = run[i:i + n]
                if gram not in freq_filtered:
                    continue
                if i > 0:
                    left.setdefault(gram, {})
                    left[gram][run[i - 1]] = left[gram].get(run[i - 1], 0) + 1
                if i + n < len(run):
                    right.setdefault(gram, {})
                    right[gram][run[i + n]] = right[gram].get(run[i + n], 0) + 1

        for gram, freq in counts.items():
            if gram not in freq_filtered or gram in DOMAIN_WORDS:
                continue
            # 互信息：取所有切分点中的最小值（最保守估计）
            pmi = float("inf")
            for k in range(1, n):
                left_count = counts_by_len[k].get(gram[:k], 0)
                right_count = counts_by_len[n - k].get(gram[k:], 0)
                if not left_count or not right_count:
                    pmi = 0.0
                    break
                pmi = min(pmi, math.log2(freq * total_chars / (left_count * right_count)))
            if pmi < min_pmi:
                continue
            if min(_entropy(left.get(gram, {})), _entropy(right.get(gram, {}))) < min_entropy:
                continue
            candidates.append((freq, gram))

    candidates.sort(key=lambda item: (-item[0], -len(item[1])))
    return [gram for _, gram in candidates[:limit]]
