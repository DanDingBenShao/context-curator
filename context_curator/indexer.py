"""IndexStore — 压缩段落的原文索引存储"""

import sqlite3
import uuid
from typing import Optional, List


class IndexStore:
    """原文索引 — 压缩后原文存此处, 需要时可拉回"""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS original_index (
                id TEXT PRIMARY KEY,
                segment_id TEXT NOT NULL,
                original_content TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_turn INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_orig_seg ON original_index(segment_id)
        """)
        self.conn.commit()

    def store(self, segment_id: str, original: str, summary: str,
              turn: int = 0) -> str:
        idx_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO original_index (id, segment_id, original_content, summary, created_turn)"
            " VALUES (?, ?, ?, ?, ?)",
            (idx_id, segment_id, original, summary, turn),
        )
        self.conn.commit()
        return idx_id

    def retrieve(self, index_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT original_content FROM original_index WHERE id = ?",
            (index_id,),
        ).fetchone()
        return row["original_content"] if row else None

    def retrieve_by_segment(self, segment_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT original_content FROM original_index WHERE segment_id = ?"
            " ORDER BY created_turn DESC LIMIT 1",
            (segment_id,),
        ).fetchone()
        return row["original_content"] if row else None

    def cleanup_orphans(self, active_segment_ids: List[str]):
        """清理没有对应活跃段落的索引"""
        if not active_segment_ids:
            self.conn.execute("DELETE FROM original_index")
            self.conn.commit()
            return

        placeholders = ",".join("?" * len(active_segment_ids))
        self.conn.execute(
            f"DELETE FROM original_index WHERE segment_id NOT IN ({placeholders})",
            active_segment_ids,
        )
        self.conn.commit()
