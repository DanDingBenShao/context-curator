"""Context Curator — 上下文管理中间层 类型定义"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Any


@dataclass
class Segment:
    """上下文段落"""
    id: str
    content: str
    score: int = 0              # 长期分 0~20
    short_term_score: int = 5   # 短期分 0~10
    marked: bool = False
    pinned: bool = False
    source_type: str = "unknown"
    compressed: bool = False
    original_idx: Optional[str] = None
    created_turn: int = 0
    last_updated_turn: int = 0


@dataclass
class ScoreAdjustment:
    """增量调分指令 — 两维独立调整"""
    segment_id: str
    delta: int = 0              # 长期分增量
    short_term_delta: int = 0   # 短期分增量


@dataclass
class InitialScore:
    """初始打分指令 (新段落) — 两维独立打分"""
    segment_id: str
    score: int = 10             # 长期初始分
    short_term_score: int = 5   # 短期初始分


@dataclass
class Compression:
    """压缩指令"""
    segment_id: str
    summary: str


@dataclass
class KnowledgeGap:
    """知识缺口"""
    type: str   # search | ask_user | memory | local_file | none
    query: str = ""
    queries: List[str] = field(default_factory=list)  # 多query: [主query, 同义变体1, ...]
    reason: str = ""


@dataclass
class GapResult:
    """知识缺口执行结果"""
    content: str
    source_type: str  # search_result | pending_ask | memory_recall | local_file


@dataclass
class CuratorOutput:
    """LLM 输出解析结果"""
    score_adjustments: List[ScoreAdjustment] = field(default_factory=list)
    initial_scores: List[InitialScore] = field(default_factory=list)
    compressions: List[Compression] = field(default_factory=list)
    knowledge_gaps: List[KnowledgeGap] = field(default_factory=list)
    strategy_update: Optional[str] = None
    delete_segments: List[str] = field(default_factory=list)
    pinned_segments: List[str] = field(default_factory=list)
    persist_segments: List[str] = field(default_factory=list)


@dataclass
class CuratorStats:
    """上下文统计"""
    total_segments: int = 0
    active_segments: int = 0
    dormant_segments: int = 0
    estimated_tokens: int = 0
    score_distribution: dict = field(default_factory=dict)
    compressed_count: int = 0
    deleted_this_turn: int = 0


@dataclass
class CuratorResult:
    """curate() 返回结果"""
    context: List[Segment]
    gap_actions: List[GapResult] = field(default_factory=list)
    skipped: bool = False
    stats: CuratorStats = field(default_factory=CuratorStats)
    pending_question: Optional[str] = None

    def has_pending_ask(self) -> bool:
        return self.pending_question is not None


@dataclass
class CuratorConfig:
    """中间层配置"""
    # 必须
    llm_client: Any = None  # LLM 客户端, 需支持 chat(messages) → str

    # 存储
    db_path: str = "curator.db"
    log_path: str = ""             # JSONL 日志路径, 空则不记录

    # TTL 参数
    decay_per_turn: int = 1
    short_term_decay: int = 2      # 短期分衰减更快
    short_term_max: int = 10       # 短期分上限
    dormant_threshold: int = 3     # 短期分 ≤ 此值为休眠
    dormant_pool_limit: int = 20   # 休眠池上限, 超出删最低长期分
    ttl_buffer: int = 2
    max_score: int = 20
    min_score: int = 0

    # 上下文限制
    max_context_tokens: int = 80000
    chars_per_token: float = 4.0
    host_model: str = ""         # 宿主模型名, 用于自动推算窗口大小
    host_reserved_tokens: int = 15000  # system prompt + tools 等固定开销

    # 复杂度跳过
    skip_below_chars: int = 20
    skip_below_segments: int = 3
    skip_below_tokens: int = 2000
    skip_patterns: List[str] = field(default_factory=lambda: [
        "fix typo",
        "fix the typo",
        "what does this variable",
        "rename",
        "add comment",
        "add a comment",
        "format this",
        "lint this",
    ])

    # 知识缺口回调 (可选, 由使用者注入)
    search_fn: Optional[Callable[[str], str]] = None
    memory_fn: Optional[Callable[[List[str]], str]] = None  # queries → 召回文本
    file_read_fn: Optional[Callable[[str], str]] = None
    on_ask_user: Optional[Callable[[str], str]] = None
    host_info_fn: Optional[Callable[[], str]] = None  # 返回宿主模型信息 JSON
