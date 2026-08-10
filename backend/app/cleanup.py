"""产物与任务记录的 TTL 清理（防磁盘涨满）。"""
from __future__ import annotations

import asyncio
import os

from app.config import settings
from app.observability import logger
from app.store import store


def _remove_outputs(state) -> None:
    if state.outputs is None:
        return
    for path in (state.outputs.glb, state.outputs.obj):
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError as e:
                logger.warning("删除产物失败 %s: %s", path, e)


def sweep_once() -> int:
    """清理一轮过期任务，返回清理数量。"""
    expired = store.expired(settings.task_ttl_seconds)
    for state in expired:
        _remove_outputs(state)
        store.delete(state.task_id)
    if expired:
        logger.info("清理过期任务 %d 个", len(expired))
    return len(expired)


async def cleanup_loop(stop: asyncio.Event) -> None:
    """周期性清理，直到收到停止信号。"""
    interval = max(30, settings.cleanup_interval_seconds)
    while not stop.is_set():
        try:
            sweep_once()
        except Exception as e:  # noqa: BLE001 - 清理失败不应影响服务
            logger.warning("清理任务异常: %s", e)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
