"""LLM prompt 模板 — 上下文管理中间层的评分/压缩/缺口分析指令"""

import json
from typing import List
from .types import Segment, CuratorOutput, ScoreAdjustment, InitialScore, Compression, KnowledgeGap


def build_curator_prompt(
    segments: List[Segment],
    strategy: str,
    user_message: str,
    turn: int,
    current_tokens: int = 0,
    max_tokens: int = 80000,
    dormant_threshold: int = 3,
    long_term_decay: int = 1,
    short_term_decay: int = 2,
) -> str:
    """构建中间层 LLM 的完整 prompt"""

    active = [s for s in segments if s.score > 0 and s.short_term_score > dormant_threshold and s.marked]
    dormant = [s for s in segments if s.score > 0 and s.short_term_score <= dormant_threshold and s.marked]
    unmarked = [s for s in segments if not s.marked and s.score > 0]

    active_lines = _format_segments(active, "活跃段落 (在宿主上下文中)")
    dormant_lines = _format_dormant_segments(dormant, "休眠段落 (不在宿主上下文, 可唤醒)")
    unmarked_lines = _format_unmarked_segments(unmarked, "未打分段落 (需给两维初始分)")

    strategy_text = strategy if strategy else "(尚无策略, 请根据本轮对话建立初始策略)"

    max_tokens = max(max_tokens, 500)
    usage_pct = current_tokens / max_tokens * 100 if max_tokens > 0 else 0
    if usage_pct > 80:
        pressure = "严重不足 — 加减分上限放宽到 ±5, 优先压缩长段落释放空间, 其次淘汰低价值段落"
    elif usage_pct > 60:
        pressure = "偏紧 — 优先保留核心信息, 对长段落考虑压缩, 低价值段落可给负分"
    elif usage_pct > 40:
        pressure = "适中 — 正常调分, 有区分度即可, 长段落可选择性压缩"
    else:
        pressure = "充裕 — 可轻减或不动, 明显噪音才给负分"

    return f"""你是上下文管理中间层。你的职责是在每轮用户发言后, 审视当前上下文并做出管理决策。

## 双维评分体系

每个段落有两个独立分数, 你负责评分, 脚本负责衰减/休眠/删除:

- **长期分 (0~20)**: 该段落在后续任务中的价值
  1. 理解基础: 缺少它会导致后续任务被误解或关键背景缺失
  2. 直接复用: 后续任务会直接引用、查询或继续讨论
  → 长期分归零 = 永久删除 (脚本执行)

- **短期分 (0~10)**: 该段落下几轮被用到的概率
  → 短期分 ≤ {dormant_threshold} 且长期分 > 0 = 休眠 (脚本执行, 不进宿主上下文但保留)
  → 想唤醒休眠段落: 给它正 short_term_delta, 让短期分超过 {dormant_threshold}

## 当前记忆管理策略
{strategy_text}

## 上下文预算
当前占用: ~{current_tokens} tokens / 上限: {max_tokens} tokens ({usage_pct:.0f}%)
预算状态: {pressure}

## 本轮用户消息 (第 {turn} 轮)
{user_message}

## 上下文段落

{active_lines}

{dormant_lines}

{unmarked_lines}

## 你的任务

请依次完成以下步骤, 以严格的 JSON 格式输出:

### Step 0: 审视需求 & 知识缺口

请按以下推理链逐步思考:

**0a. 分析用户意图**
用户要主 Agent 做什么? 是继续开发、回答技术问题、排查故障, 还是闲聊?

**0b. 拆解前置知识**
要完成这个任务, 主 Agent 需要知道哪些背景? 思考:
- 项目目标/阶段: 这个项目的最终目标是什么? 当前处于什么阶段?
- 技术约束/决策: 有没有关键的技术选型、架构决策、编码规范?
- 历史背景: 之前讨论过什么相关话题? 达成了什么共识?
- 用户偏好: 用户对方案选择、代码风格有什么倾向?

**0c. 缺口中识别**
逐一对照已有上下文段落——上述前置知识中, 哪些在当前上下文中缺失或不够?
→ 如果已有段落充分覆盖, 无需额外检索。
→ 如果有缺口且属于跨会话长期记忆 (之前的项目决策、用户偏好、历史共识等), 在 `knowledge_gaps` 中发起 `memory` 检索。
→ 如果需要最新外部信息 (API 文档、版本更新等), 发起 `search` 检索。
→ 如果需要用户澄清意图或补充背景, 发起 `ask_user`。
→ 当前上下文已足够 → type: "none"。

**0d. 策略更新**
用户的关注焦点是否变化? 如有变化, 更新记忆管理策略作为后续评分的宏观指引。
同时检查用户是否明确要求"记住"/"不要忘记"/"这条很重要"等——如有, 将对应段落的 id 加入 `pinned_segments`。

输出字段: `strategy_update` (null 或新策略文本), `pinned_segments`, `knowledge_gaps`

### Step 1: 评分
长期分和短期分是独立的两个维度——重要但不紧急的信息可以长期分高、短期分低(休眠), 反之亦然。
注意: 脚本每轮自动衰减(长期-{long_term_decay}, 短期-{short_term_decay}), 如果段落价值并未降低, 你应给正分抵消衰减; 只有价值真的下降了才不抵消或给负分。

- **未打分段落**: 给出 long_term_score (0~20) 和 short_term_score (0~10)
  - 长期 20 = 核心约束/用户铁律/项目定义
  - 长期 15 = 重要决策或背景
  - 长期 10 = 有一定参考价值
  - 长期 5 = 仅当轮有用, 后续不再需要
  - 长期 0 = 完全无用, 应删除
  - 短期 8~10 = 下轮马上用到
  - 短期 5~7 = 接下来几轮可能用到
  - 短期 0~4 = 短期用不到, 可休眠
- **活跃已打分段落**: 根据当前需求变化和预算状态 ({pressure}) 做增量调整
  - long_term_delta 和 short_term_delta, 单次上限 ±3, 严重不足时 ±5
  - 长期分调整后若 ≤ 0 加入 `delete_segments`
  - 短期分降低 = 让段落休眠 (如果长期分 > 0 会保留)
  - [锁] 表示用户明确要求保留, 应尽量给正分确保不被删除或休眠
- **休眠段落**: 对比本轮用户消息与休眠段落内容, 如果话题重新关联则唤醒——给正 short_term_delta 使其 > {dormant_threshold}
  - 休眠段落已列在下方, 通过 score_adjustments 给正 short_term_delta 即可唤醒

输出字段: `initial_scores`, `score_adjustments`, `delete_segments`

### Step 2: 压缩
对工具输出/代码块等长段落, 用摘要替换原文以节省上下文空间。原文存入外部索引, 需要时可拉回。
压缩比删除更好——既保留信息又释放预算。预算越紧张, 越应优先压缩长段落。
输出字段: `compressions`

## 输出格式

严格输出以下 JSON, 不要包含任何其他文本:

```json
{{
  "strategy_update": null,
  "pinned_segments": [],
  "initial_scores": [
    {{"segment_id": "xxx", "long_term_score": 15, "short_term_score": 8}}
  ],
  "score_adjustments": [
    {{"segment_id": "yyy", "long_term_delta": -2, "short_term_delta": -3}}
  ],
  "delete_segments": ["aaa"],
  "compressions": [
    {{"segment_id": "ccc", "summary": "讨论了连接池配置, 决定使用 pgbouncer"}}
  ],
  "knowledge_gaps": [
    {{"type": "none", "query": "", "reason": ""}}
  ]
}}
```

注意:
- 每轮衰减自动发生, 价值未变的段落需要正分抵消衰减, 不要被动等 decay 降分
- 未打分段落必须全部给分, 不能遗漏
- 休眠段落通过 score_adjustments 给正 short_term_delta 即可唤醒
- 预算紧张时, 压缩优于删除——既保留信息又释放空间
"""


def _format_segments(segments: List[Segment], title: str) -> str:
    if not segments:
        return f"### {title}\n(无)"

    lines = [f"### {title}\n"]
    for seg in segments:
        tag = ""
        if seg.compressed:
            tag += "[压]"
        if seg.pinned:
            tag += "[锁]"
        content_preview = seg.content[:500] + ("..." if len(seg.content) > 500 else "")
        lines.append(
            f"[ID:{seg.id}] [长期:{seg.score}] [短期:{seg.short_term_score}] [源:{seg.source_type}] {tag}\n"
            f"{content_preview}\n"
        )
    return "\n".join(lines)


def _format_dormant_segments(segments: List[Segment], title: str) -> str:
    """休眠段落格式化"""
    if not segments:
        return f"### {title}\n(无)"

    lines = [f"### {title}\n"]
    for seg in segments:
        tag = "[睡]"
        if seg.compressed:
            tag += "[压]"
        if seg.pinned:
            tag += "[锁]"
        content_preview = seg.content[:300] + ("..." if len(seg.content) > 300 else "")
        lines.append(
            f"[ID:{seg.id}] {tag} [长期:{seg.score}] [短期:{seg.short_term_score}] [源:{seg.source_type}]\n"
            f"{content_preview}\n"
        )
    return "\n".join(lines)


def _format_unmarked_segments(segments: List[Segment], title: str) -> str:
    if not segments:
        return f"### {title}\n(无)"
    lines = [f"### {title}\n"]
    for seg in segments:
        tag = ""
        if seg.compressed:
            tag += "[压]"
        if seg.pinned:
            tag += "[锁]"
        content_preview = seg.content[:500] + ("..." if len(seg.content) > 500 else "")
        lines.append(
            f"[ID:{seg.id}] [源:{seg.source_type}] {tag}\n"
            f"{content_preview}\n"
        )
    return "\n".join(lines)


def _repair_json(text: str) -> str:
    """修复 LLM 常见的 JSON 格式错误"""
    import re
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r'"\s+"', '", "', text)
    text = re.sub(r'\}\s*\{', '}, {', text)
    text = re.sub(r'\]\s*\[', '], [', text)
    text = re.sub(r':\s*"([^"]*?)\s*\n', r': "\1"\n', text)
    return text


def parse_curator_output(raw: str) -> CuratorOutput:
    """解析 LLM 输出的 JSON, 容错处理"""
    text = raw.strip()
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.rindex("```")
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.rindex("```")
        text = text[start:end].strip()

    data = None
    for attempt in range(3):
        try:
            data = json.loads(text)
            break
        except json.JSONDecodeError:
            if attempt == 0:
                text = _repair_json(text)
            elif attempt == 1:
                import re
                match = re.search(r'\{[\s\S]*\}', text)
                if match:
                    text = _repair_json(match.group())
                else:
                    break

    if data is None:
        return CuratorOutput()

    return CuratorOutput(
        strategy_update=data.get("strategy_update"),
        initial_scores=[
            InitialScore(
                segment_id=s["segment_id"],
                score=s.get("long_term_score", s.get("score", 10)),
                short_term_score=s.get("short_term_score", 5),
            )
            for s in data.get("initial_scores", [])
        ],
        score_adjustments=[
            ScoreAdjustment(
                segment_id=a["segment_id"],
                delta=a.get("long_term_delta", a.get("delta", 0)),
                short_term_delta=a.get("short_term_delta", 0),
            )
            for a in data.get("score_adjustments", [])
        ],
        delete_segments=data.get("delete_segments", []),
        pinned_segments=data.get("pinned_segments", []),
        compressions=[
            Compression(segment_id=c["segment_id"], summary=c["summary"])
            for c in data.get("compressions", [])
        ],
        knowledge_gaps=[
            KnowledgeGap(
                type=g.get("type", "none"),
                query=g.get("query", ""),
                reason=g.get("reason", ""),
            )
            for g in data.get("knowledge_gaps", [])
        ],
    )
