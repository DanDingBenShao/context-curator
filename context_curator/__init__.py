"""Context Curator — 上下文管理中间层

为 LLM Agent 提供可插拔的上下文生命周期管理:
  - 双维评分: 长期分 (0~20, 后续任务价值) + 短期分 (0~10, 近期用到概率)
  - 休眠唤醒: 短期分低于阈值自动休眠, 不进宿主上下文但保留, 话题回归可唤醒
  - 压缩索引: LLM 压缩长段落为摘要, 原文存入外部索引可拉回
  - Pin 保护: [锁] 标记用户明确要求保留的内容
  - 预算感知: 四级压力自适应调整评分/压缩/淘汰策略
  - 知识缺口检测: Step 0 推理链 — 分析意图 → 拆解前置知识 → 对照上下文 → 发现缺失发起检索
  - 多 query 记忆召回: LLM 生成主 query + 同义变体, 提高召回覆盖率; 单 query 系统取 queries[0] 兜底
  - 脚本执行: LLM 负责评分和缺口判断, 衰减/休眠/删除/淘汰由脚本执行

使用:
    from context_curator import create_curator

    curator = create_curator(
        max_tokens=80000,
        memory_fn=lambda queries: mem0.search(queries),
    )
    result = curator.curate(user_message)
    clean_context = result.context  # 传给主 Agent
"""

import os
from typing import Optional

from .types import (
    CuratorConfig,
    CuratorResult,
    CuratorStats,
    CuratorOutput,
    Segment,
    GapResult,
    KnowledgeGap,
    ScoreAdjustment,
    InitialScore,
    Compression,
)
from .curator import ContextCurator

__version__ = "0.2.0"
__all__ = [
    "ContextCurator",
    "CuratorConfig",
    "CuratorResult",
    "CuratorStats",
    "Segment",
    "GapResult",
    "KnowledgeGap",
    "create_curator",
]


def create_curator(
    api_key: str = "",
    api_base: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    db_path: str = "curator.db",
    max_tokens: int = 80000,
    host_model: str = "",
    log_path: str = "",
    **kwargs,
) -> ContextCurator:
    """一键创建 ContextCurator, 零配置接入任意 Agent。

    Args:
        api_key: DeepSeek-compatible API key, 默认读环境变量 CURATOR_API_KEY
        api_base: API 地址, 默认 https://api.deepseek.com
        model: curator 用的小模型, 默认 deepseek-chat
        db_path: SQLite 持久化路径
        max_tokens: 上下文预算上限
        host_model: 宿主模型名 (自动推算窗口), 如 "claude-sonnet-4-6"
        log_path: JSONL 日志路径, 空则不记录
        **kwargs: 传给 CuratorConfig 的其他参数

    Returns:
        配置好的 ContextCurator 实例, 直接调用 curate() 即可
    """
    api_key = api_key or os.environ.get("CURATOR_API_KEY", "")
    if not api_key:
        raise ValueError(
            "请提供 api_key 或设置环境变量 CURATOR_API_KEY"
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "需要安装 openai: pip install openai"
        )

    client = OpenAI(api_key=api_key, base_url=api_base)

    def _llm(prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0, max_tokens=2000,
        )
        return resp.choices[0].message.content

    config = CuratorConfig(
        llm_client=_llm,
        db_path=db_path,
        max_context_tokens=max_tokens,
        host_model=host_model,
        log_path=log_path,
        **kwargs,
    )
    return ContextCurator(config)
