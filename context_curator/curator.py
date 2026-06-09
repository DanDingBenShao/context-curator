"""ContextCurator — 上下文管理中间层主编排器

每轮用户发言后调用一次, 完成:
  0. 需求审视 → 更新记忆管理策略
  1. 段落评分 (新段落初始分 + 已有段落增量调分)
  2. 压缩 (摘要留 context + 原文存索引)
  3. 知识缺口 (搜/问/查/读)
  4. 末尾淘汰兜底
  5. 组装干净 context 返回给主 Agent
"""

import json
from typing import List, Optional

from .types import (
    CuratorConfig, CuratorResult, CuratorStats,
    Segment, GapResult, KnowledgeGap, CuratorOutput,
)
from .segment import SegmentStore
from .indexer import IndexStore
from .gate import ComplexityGate
from .prompts import build_curator_prompt, parse_curator_output

# 常见模型上下文窗口 (token)
MODEL_WINDOWS = {
    "claude-opus-4-7":     200000,
    "claude-sonnet-4-6":   200000,
    "claude-haiku-4-5":    200000,
    "deepseek-v4-pro":    1000000,
    "deepseek-v4-flash":   128000,
    "deepseek-chat":       128000,
    "deepseek-v3":         128000,
    "deepseek-r1":         128000,
    "gpt-4o":              128000,
    "gpt-4-turbo":         128000,
    "gpt-4":                8192,
    "gemini-2.5-pro":     1048576,
    "qwen-max":            131072,
}


class ContextCurator:
    """上下文管理中间层 — 可插入任意 Agent 的插件"""

    def __init__(self, config: CuratorConfig):
        self.config = config
        self.segments = SegmentStore(config.db_path)
        self.indexer = IndexStore(config.db_path)
        self.gate = ComplexityGate(config)
        self.strategy: str = ""
        self.turn: int = 0
        self._host_window: int = 0  # 0=未检测, -1=已检测但无宿主信息
        self._host_checked: bool = False
        self._user_max_tokens: int = config.max_context_tokens  # 记录原始值

    def curate(self, user_message: str) -> CuratorResult:
        """每轮用户发言后调用, 返回干净上下文和缺口动作"""
        self.turn += 1
        dt = self.config.dormant_threshold

        # ── 1. 脚本 decay (先衰减旧段落, 新消息不受影响) ──
        self.segments.decay_all(self.config.decay_per_turn)
        self.segments.decay_short_term(self.config.short_term_decay)

        # ── 2. 添加本轮用户消息 (双分默认值, 不被首轮衰减) ──
        self.segments.add(user_message, "user_message", self.turn,
                          score=10, short_term_score=5)

        # ── 2b. 检测宿主窗口 (首次或模型变更后) ──
        if self._host_window == 0 and not self._host_checked:
            self._detect_host()

        # ── 3. 复杂度判断 ──
        if self.gate.should_skip(user_message, self.segments,
                                 self.config.chars_per_token, dt):
            stats = self.segments.get_stats(self.config.chars_per_token, dt)
            self._write_log(turn=self.turn, user_message=user_message,
                           current_tokens=0, max_tokens=self.config.max_context_tokens,
                           parsed=None, stats=stats, skipped=True)
            return CuratorResult(
                context=self.segments.get_active_for_host(dt),
                skipped=True,
                stats=stats,
            )

        # ── 4. 预算感知: 计算当前 token 用量 (仅活跃, 休眠不占宿主预算) ──
        active = self.segments.get_active_for_host(dt)
        current_chars = sum(len(s.content) for s in active)
        current_tokens = int(current_chars / self.config.chars_per_token)

        # ── 5. 调用 LLM 评分/压缩/缺口分析 ──
        prompt = build_curator_prompt(
            segments=list(self.segments.get_all()),
            strategy=self.strategy,
            user_message=user_message,
            turn=self.turn,
            current_tokens=current_tokens,
            max_tokens=self.config.max_context_tokens,
            dormant_threshold=dt,
            long_term_decay=self.config.decay_per_turn,
            short_term_decay=self.config.short_term_decay,
        )
        llm_output = self._call_llm(prompt)
        parsed = parse_curator_output(llm_output)

        # ── 成本追踪: Curator 自身消耗 ──
        cp = self.config.chars_per_token
        curator_prompt_tokens = int(len(prompt) / cp)
        curator_resp_tokens = int(len(llm_output) / cp) if llm_output else 0

        # ── 节省追踪: 删除前快照 ──
        deletion_saved = 0
        for seg_id in parsed.delete_segments:
            seg = self.segments.get(seg_id)
            if seg:
                deletion_saved += int(len(seg.content) / cp)

        # ── 5. 应用删除 ──
        for seg_id in parsed.delete_segments:
            self.segments.delete(seg_id)

        # ── 6. 应用评分变更 (两维) ──
        if parsed.initial_scores:
            self.segments.set_initial_scores(parsed.initial_scores, self.turn)
        if parsed.score_adjustments:
            self.segments.apply_dual_deltas(parsed.score_adjustments, self.turn)

        # ── 6b. 处理 LLM 识别的 pinned 段落 ──
        for seg_id in parsed.pinned_segments:
            self.segments.pin_segment(seg_id)

        # ── 保存追踪: 压缩前快照 ──
        compression_saved = 0

        # ── 7. 执行压缩 ──
        for comp in parsed.compressions:
            seg = self.segments.get(comp.segment_id)
            if seg is None:
                continue
            orig_tokens = int(len(seg.content) / cp)
            summary_tokens = int(len(comp.summary) / cp)
            compression_saved += max(0, orig_tokens - summary_tokens)
            idx_id = self.indexer.store(
                seg.id, seg.content, comp.summary, self.turn
            )
            self.segments.compress(seg.id, comp.summary, idx_id)

        # ── 8. 更新策略 ──
        if parsed.strategy_update:
            self.strategy = parsed.strategy_update

        # ── 9. 执行知识缺口 ──
        gap_results = self._execute_gaps(parsed.knowledge_gaps)
        for gr in gap_results:
            self.segments.add(gr.content, gr.source_type, self.turn,
                              score=10, short_term_score=7)

        # ── 10. 末尾淘汰兜底 (仅从活跃段落中删) ──
        self.segments.trim_by_tokens(
            max(self.config.max_context_tokens, 500),
            self.config.chars_per_token,
        )

        # ── 10b. 休眠池上限 ──
        self.segments.trim_dormant_pool(
            self.config.dormant_pool_limit, dt
        )

        # ── 11. 清理孤儿索引 ──
        active_ids = [s.id for s in self.segments.get_active()]
        self.indexer.cleanup_orphans(active_ids)

        # ── 12. 组装返回 (只给宿主活跃段落) ──
        pending = _extract_pending_ask(gap_results)
        final_stats = self.segments.get_stats(self.config.chars_per_token, dt)

        # ── 成本汇总 ──
        # 休眠节省: 休眠段落不传给宿主, 省下的 token
        dormant = self.segments.get_dormant(dt)
        dormant_saved = sum(int(len(s.content) / cp) for s in dormant)

        total_saved = deletion_saved + compression_saved + dormant_saved
        curator_spent = curator_prompt_tokens + curator_resp_tokens

        cost = {
            "saved": {
                "deletion": deletion_saved,
                "compression": compression_saved,
                "dormancy": dormant_saved,
                "total": total_saved,
            },
            "spent": {
                "prompt": curator_prompt_tokens,
                "response": curator_resp_tokens,
                "total": curator_spent,
            },
            "net": total_saved - curator_spent,
        }

        # ── 13. 写日志 ──
        self._write_log(
            turn=self.turn,
            user_message=user_message,
            current_tokens=current_tokens,
            max_tokens=self.config.max_context_tokens,
            parsed=parsed,
            stats=final_stats,
            skipped=False,
            cost=cost,
        )

        return CuratorResult(
            context=self.segments.get_active_for_host(dt),
            gap_actions=gap_results,
            skipped=False,
            stats=final_stats,
            pending_question=pending,
        )

    # ── 知识缺口执行 ──

    def _execute_gaps(self, gaps: List[KnowledgeGap]) -> List[GapResult]:
        results = []
        seen_queries = set()
        for gap in gaps:
            if gap.type == "search" and self.config.search_fn:
                try:
                    content = self.config.search_fn(gap.query)
                    results.append(GapResult(content, "search_result"))
                except Exception as e:
                    results.append(GapResult(
                        f"[搜索失败] {gap.query}: {e}", "search_result"
                    ))
            elif gap.type == "ask_user":
                results.append(GapResult(
                    f"[待用户确认] {gap.query}", "pending_ask"
                ))
            elif gap.type == "memory" and self.config.memory_fn:
                # 组装完整 query 列表, 去重避免连续两轮相同 gap 重复调用
                queries = [gap.query] if gap.query else []
                queries += gap.queries
                key = json.dumps(queries, ensure_ascii=False)
                if key in seen_queries:
                    continue
                seen_queries.add(key)
                try:
                    content = self.config.memory_fn(queries if queries else [])
                    results.append(GapResult(content, "memory_recall"))
                except Exception as e:
                    results.append(GapResult(
                        f"[记忆查询失败] queries={queries}: {e}", "memory_recall"
                    ))
            elif gap.type == "local_file" and self.config.file_read_fn:
                try:
                    content = self.config.file_read_fn(gap.query)
                    results.append(GapResult(content, "local_file"))
                except Exception as e:
                    results.append(GapResult(
                        f"[文件读取失败] {gap.query}: {e}", "local_file"
                    ))
            elif gap.type == "none":
                pass
            elif gap.type == "host_info":
                info = self._resolve_host_info()
                if info:
                    self._apply_host_info(info)
                    results.append(GapResult(
                        f"[宿主环境已更新] {json.dumps(info, ensure_ascii=False)}",
                        "host_info"
                    ))
        return results

    # ── 宿主环境检测 ──

    def _detect_host(self):
        """检测宿主模型窗口, 注入为上下文段落。无宿主信息时静默跳过。"""
        info = self._resolve_host_info()
        if info:
            self._apply_host_info(info)
            content = json.dumps(info, ensure_ascii=False)
            new_seg_content = f"[宿主环境] {content}"
            # 检查是否与已有 host_info 相同, 相同则跳过
            for s in self.segments.get_all():
                if s.source_type == "host_info":
                    if s.content == new_seg_content:
                        self._host_checked = True
                        return
                    self.segments.delete(s.id)
            self.segments.add(
                new_seg_content, "host_info", self.turn,
                score=20, short_term_score=10, pinned=True
            )
        else:
            self._host_window = -1  # 标记已检测, 避免重复
        self._host_checked = True

    def _apply_host_info(self, info: dict):
        """应用宿主管信息, 更新预算上限"""
        self._host_window = info.get("context_window", 0)
        reserved = self.config.host_reserved_tokens
        effective = max(1000, self._host_window - reserved)
        if self._user_max_tokens <= 0:
            # 用户未设上限, 自动跟随宿主
            self.config.max_context_tokens = effective
        else:
            # 用户设了上限, 取 min(用户上限, 宿主能力)
            self.config.max_context_tokens = min(self._user_max_tokens, effective)

    def notify_model_change(self):
        """宿主模型变更时调用, 下轮 curate 会重新检测"""
        self._host_window = 0
        self._host_checked = False

    def _resolve_host_info(self) -> Optional[dict]:
        """解析宿主信息: host_info_fn 回调 > host_model 查表 > None"""
        if self.config.host_info_fn:
            try:
                result = self.config.host_info_fn()
                if isinstance(result, str):
                    return json.loads(result)
                return result
            except Exception:
                pass
        if self.config.host_model:
            model_lower = self.config.host_model.lower()
            window = MODEL_WINDOWS.get(model_lower)
            if window:
                return {"model": self.config.host_model, "context_window": window}
        return None

    # ── 日志 ──

    def _write_log(self, turn: int, user_message: str, current_tokens: int,
                   max_tokens: int, parsed, stats, skipped: bool,
                   cost: Optional[dict] = None):
        """写 JSONL 日志, 记录每轮决策和 token 成本"""
        if not self.config.log_path:
            return

        entry = {
            "turn": turn,
            "user_message": user_message[:200],
            "budget": {
                "current_tokens": current_tokens,
                "max_tokens": max_tokens,
                "pct": round(current_tokens / max(max_tokens, 1) * 100),
            },
            "skipped": skipped,
        }

        if parsed is not None:
            entry["llm"] = {
                "strategy_update": parsed.strategy_update,
                "pinned": parsed.pinned_segments,
                "initial_scored": len(parsed.initial_scores),
                "adjustments": len(parsed.score_adjustments),
                "deleted": parsed.delete_segments,
                "compressed": [{"id": c.segment_id, "summary": c.summary[:100]} for c in parsed.compressions],
                "knowledge_gaps": [{"type": g.type, "query": g.query, "queries": g.queries} for g in parsed.knowledge_gaps if g.type != "none"],
                "persisted": parsed.persist_segments,
            }

        entry["result"] = {
            "active": stats.active_segments,
            "dormant": stats.dormant_segments,
            "total": stats.total_segments,
            "tokens": stats.estimated_tokens,
            "compressed_total": stats.compressed_count,
            "score_distribution": stats.score_distribution,
        }

        if cost is not None:
            entry["cost"] = cost

        import json
        with open(self.config.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── LLM 调用 ──

    def _call_llm(self, prompt: str) -> str:
        client = self.config.llm_client
        if client is None:
            return '{"strategy_update": null, "pinned_segments": [], "initial_scores": [], "score_adjustments": [], "delete_segments": [], "compressions": [], "knowledge_gaps": [{"type": "none", "query": "", "queries": [], "reason": ""}], "persist_segments": []}'

        # 支持三种接口: callable(prompt) / client.chat(prompt) / client.chat(messages)
        if callable(client) and not hasattr(client, "chat"):
            return client(prompt)

        try:
            return client.chat(prompt)
        except (TypeError, AttributeError):
            return client.chat([{"role": "user", "content": prompt}])

    # ── 原文拉回 ──

    def pull_original(self, segment_id: str) -> Optional[str]:
        """根据段落 ID 拉回原文"""
        return self.indexer.retrieve_by_segment(segment_id)

    def pull_by_index(self, index_id: str) -> Optional[str]:
        """根据索引 ID 拉回原文"""
        return self.indexer.retrieve(index_id)


def _extract_pending_ask(gap_results: List[GapResult]) -> Optional[str]:
    for gr in gap_results:
        if gr.source_type == "pending_ask":
            return gr.content
    return None


