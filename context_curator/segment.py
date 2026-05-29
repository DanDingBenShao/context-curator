"""SegmentStore — 上下文段落的 SQLite 持久化与生命周期管理"""

import sqlite3
import uuid
import math
from typing import List, Optional, Dict

from .types import Segment, ScoreAdjustment, InitialScore, CuratorStats


class SegmentStore:
    """段落存储 — SQLite + TTL 衰减 + 末尾淘汰"""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                score INTEGER NOT NULL DEFAULT 0,
                short_term_score INTEGER NOT NULL DEFAULT 5,
                marked INTEGER NOT NULL DEFAULT 0,
                pinned INTEGER NOT NULL DEFAULT 0,
                source_type TEXT NOT NULL DEFAULT 'unknown',
                compressed INTEGER NOT NULL DEFAULT 0,
                original_idx TEXT,
                created_turn INTEGER NOT NULL DEFAULT 0,
                last_updated_turn INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Add pinned column for existing databases
        try:
            self.conn.execute("ALTER TABLE segments ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        except:
            pass
        # Add short_term_score column for existing databases
        try:
            self.conn.execute("ALTER TABLE segments ADD COLUMN short_term_score INTEGER NOT NULL DEFAULT 5")
        except:
            pass
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seg_score ON segments(score)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seg_marked ON segments(marked)
        """)
        self.conn.commit()

    # ── CRUD ──

    def add(self, content: str, source_type: str = "unknown", turn: int = 0,
            score: int = 0, short_term_score: int = 5, pinned: bool = False) -> str:
        seg_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO segments (id, content, score, short_term_score, pinned, source_type, created_turn, last_updated_turn)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (seg_id, content, score, short_term_score, int(pinned), source_type, turn, turn),
        )
        self.conn.commit()
        return seg_id

    def add_batch(self, items: List[dict], turn: int = 0) -> List[str]:
        """items: [{"content": ..., "source_type": ...}, ...]"""
        ids = []
        for item in items:
            seg_id = self.add(
                item.get("content", ""),
                item.get("source_type", "unknown"),
                turn,
                item.get("score", 0),
            )
            ids.append(seg_id)
        return ids

    def get(self, seg_id: str) -> Optional[Segment]:
        row = self.conn.execute(
            "SELECT * FROM segments WHERE id = ?", (seg_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_segment(row)

    def delete(self, seg_id: str):
        self.conn.execute("DELETE FROM segments WHERE id = ?", (seg_id,))
        self.conn.commit()

    def pin_segment(self, seg_id: str):
        """标记段落为 pinned, LLM 后续会根据 [锁] 标记倾向给高分"""
        self.conn.execute(
            "UPDATE segments SET pinned = 1 WHERE id = ?", (seg_id,)
        )
        self.conn.commit()

    # ── TTL 衰减 ──

    def decay_all(self, amount: int = 1) -> List[str]:
        """所有段落长期分 -amount, 删除 score <= 0 的段落"""
        self.conn.execute(
            "UPDATE segments SET score = score - ? WHERE score > 0", (amount,)
        )
        deleted = self.conn.execute(
            "SELECT id FROM segments WHERE score <= 0"
        ).fetchall()
        deleted_ids = [r["id"] for r in deleted]
        if deleted_ids:
            placeholders = ",".join("?" * len(deleted_ids))
            self.conn.execute(
                f"DELETE FROM segments WHERE id IN ({placeholders})",
                deleted_ids,
            )
        self.conn.commit()
        return deleted_ids

    def decay_short_term(self, amount: int = 2):
        """所有段落短期分 -amount, 不低于 0"""
        self.conn.execute(
            "UPDATE segments SET short_term_score = MAX(0, short_term_score - ?)"
            " WHERE score > 0", (amount,)
        )
        self.conn.commit()

    # ── 评分 ──

    def set_initial_scores(self, scores: List[InitialScore], turn: int = 0):
        """给新段落打初始分并标记 — 两维"""
        for s in scores:
            lt = max(0, min(20, s.score))
            st = max(0, min(10, getattr(s, 'short_term_score', 5)))
            self.conn.execute(
                "UPDATE segments SET score = ?, short_term_score = ?, marked = 1, last_updated_turn = ?"
                " WHERE id = ?",
                (lt, st, turn, s.segment_id),
            )
        self.conn.commit()

    def apply_deltas(self, adjustments: List[ScoreAdjustment], turn: int = 0):
        """对已有标记段落增量调分 (仅长期分, 向后兼容)"""
        for adj in adjustments:
            row = self.conn.execute(
                "SELECT score FROM segments WHERE id = ?", (adj.segment_id,)
            ).fetchone()
            if row is None:
                continue
            new_score = max(0, min(20, row["score"] + adj.delta))
            if new_score == 0:
                self.conn.execute(
                    "DELETE FROM segments WHERE id = ?", (adj.segment_id,)
                )
            else:
                self.conn.execute(
                    "UPDATE segments SET score = ?, last_updated_turn = ?"
                    " WHERE id = ?",
                    (new_score, turn, adj.segment_id),
                )
        self.conn.commit()

    def apply_dual_deltas(self, adjustments: List[ScoreAdjustment], turn: int = 0):
        """两维增量调分 — 长期分归零删除, 短期分归零不删"""
        for adj in adjustments:
            row = self.conn.execute(
                "SELECT score, short_term_score FROM segments WHERE id = ?",
                (adj.segment_id,)
            ).fetchone()
            if row is None:
                continue
            new_lt = max(0, min(20, row["score"] + adj.delta))
            new_st = max(0, min(10, row["short_term_score"] + adj.short_term_delta))
            if new_lt == 0:
                self.conn.execute("DELETE FROM segments WHERE id = ?", (adj.segment_id,))
            else:
                self.conn.execute(
                    "UPDATE segments SET score = ?, short_term_score = ?, last_updated_turn = ?"
                    " WHERE id = ?",
                    (new_lt, new_st, turn, adj.segment_id),
                )
        self.conn.commit()

    # ── 压缩 ──

    def compress(self, seg_id: str, summary: str, original_idx: str):
        """将段落内容替换为摘要, 标记已压缩"""
        self.conn.execute(
            "UPDATE segments SET content = ?, compressed = 1, original_idx = ?"
            " WHERE id = ?",
            (summary, original_idx, seg_id),
        )
        self.conn.commit()

    # ── 查询 ──

    def get_active(self) -> List[Segment]:
        """获取所有活跃段落 (score > 0) — curator 内部使用"""
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE score > 0 ORDER BY created_turn ASC, rowid ASC"
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def get_active_for_host(self, dormant_threshold: int = 3) -> List[Segment]:
        """获取应传给宿主的段落 (长期>0 且 短期>阈值)"""
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE score > 0 AND short_term_score > ?"
            " ORDER BY created_turn ASC, rowid ASC",
            (dormant_threshold,)
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def get_dormant(self, dormant_threshold: int = 3) -> List[Segment]:
        """获取休眠段落 (长期>0 但 短期≤阈值)"""
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE score > 0 AND short_term_score <= ?"
            " ORDER BY created_turn ASC, rowid ASC",
            (dormant_threshold,)
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def get_unmarked(self) -> List[Segment]:
        """获取未打分段落 (冷启动)"""
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE marked = 0 AND score > 0"
            " ORDER BY created_turn ASC, rowid ASC"
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def get_marked(self) -> List[Segment]:
        """获取已打分段落"""
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE marked = 1 AND score > 0"
            " ORDER BY created_turn ASC, rowid ASC"
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def get_all(self) -> List[Segment]:
        """获取所有段落 (含 score=0)"""
        rows = self.conn.execute(
            "SELECT * FROM segments ORDER BY created_turn ASC, rowid ASC"
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    # ── 末尾淘汰 ──

    def trim_by_tokens(self, max_tokens: int, chars_per_token: float = 4.0):
        """从最低分段落开始删除, 直到总 token 估算 <= max_tokens"""
        active = self.get_active()
        total_chars = sum(len(s.content) for s in active)
        estimated = total_chars / chars_per_token

        if estimated <= max_tokens:
            return

        to_remove = int((estimated - max_tokens) * chars_per_token)
        removed_chars = 0

        for seg in sorted(active, key=lambda s: s.score):
            if removed_chars >= to_remove:
                break
            self.delete(seg.id)
            removed_chars += len(seg.content)

    # ── 休眠池管理 ──

    def trim_dormant_pool(self, max_dormant: int = 20, dormant_threshold: int = 3):
        """休眠段落超过上限时, 删除最低长期分的"""
        dormant = self.get_dormant(dormant_threshold)
        if len(dormant) <= max_dormant:
            return
        to_remove = len(dormant) - max_dormant
        for seg in sorted(dormant, key=lambda s: s.score)[:to_remove]:
            self.delete(seg.id)

    # ── 统计 ──

    def get_stats(self, chars_per_token: float = 4.0, dormant_threshold: int = 3) -> CuratorStats:
        all_segs = self.get_all()
        active = [s for s in all_segs if s.score > 0]
        host_active = [s for s in active if s.short_term_score > dormant_threshold]
        dormant = [s for s in active if s.short_term_score <= dormant_threshold]
        total_chars = sum(len(s.content) for s in host_active)
        dist = {}
        for s in host_active:
            bucket = f"{s.score // 5 * 5}-{s.score // 5 * 5 + 4}"
            dist[bucket] = dist.get(bucket, 0) + 1
        return CuratorStats(
            total_segments=len(all_segs),
            active_segments=len(host_active),
            dormant_segments=len(dormant),
            estimated_tokens=int(total_chars / chars_per_token),
            score_distribution=dist,
            compressed_count=sum(1 for s in host_active if s.compressed),
        )

    # ── 辅助 ──

    def _row_to_segment(self, row) -> Segment:
        return Segment(
            id=row["id"],
            content=row["content"],
            score=row["score"],
            short_term_score=row["short_term_score"],
            marked=bool(row["marked"]),
            pinned=bool(row["pinned"]),
            source_type=row["source_type"],
            compressed=bool(row["compressed"]),
            original_idx=row["original_idx"],
            created_turn=row["created_turn"],
            last_updated_turn=row["last_updated_turn"],
        )
