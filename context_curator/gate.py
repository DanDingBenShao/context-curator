"""ComplexityGate — 判断是否跳过中间层, 直接进入主 Agent"""

from .types import CuratorConfig


class ComplexityGate:
    """复杂度判断 — 简单任务跳过中间层"""

    def __init__(self, config: CuratorConfig):
        self.skip_below_chars = config.skip_below_chars
        self.skip_below_segments = config.skip_below_segments
        self.skip_below_tokens = config.skip_below_tokens
        self.skip_patterns = [p.lower() for p in config.skip_patterns]

    def should_skip(self, user_message: str, segment_store,
                    chars_per_token: float = 4.0,
                    dormant_threshold: int = 3) -> bool:
        """满足任一条件返回 True, 跳过中间层"""
        msg = user_message.strip()

        # 1. 消息太短
        if len(msg) < self.skip_below_chars:
            return True

        # 2. 匹配简单模式
        msg_lower = msg.lower()
        for pattern in self.skip_patterns:
            if pattern in msg_lower:
                return True

        # 3. 上下文太少 (只计对宿主可见的活跃段落)
        active = segment_store.get_active_for_host(dormant_threshold)
        if len(active) < self.skip_below_segments:
            total_chars = sum(len(s.content) for s in active) + len(msg)
            if total_chars / chars_per_token < self.skip_below_tokens:
                return True

        return False
