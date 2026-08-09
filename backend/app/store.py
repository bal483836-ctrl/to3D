"""进程内任务存储与进度广播（MVP）。

生产环境应替换为 PostgreSQL(状态) + Redis(缓存/pubsub)（方案 §8）。
接口保持一致，便于替换。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from app.models.schemas import TaskState


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskState] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._decisions: dict[str, asyncio.Queue] = {}

    def create(self, state: TaskState) -> None:
        self._tasks[state.task_id] = state
        self._subscribers.setdefault(state.task_id, [])
        self._decisions.setdefault(state.task_id, asyncio.Queue())

    def get(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    async def update(self, state: TaskState) -> None:
        self._tasks[state.task_id] = state
        for q in self._subscribers.get(state.task_id, []):
            await q.put(state.model_copy(deep=True))

    def subscribe(self, task_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(task_id, []).append(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(task_id, [])
        if q in subs:
            subs.remove(q)

    # 人在环决策通道
    async def push_decision(self, task_id: str, decision) -> None:
        await self._decisions[task_id].put(decision)

    async def wait_decision(self, task_id: str):
        return await self._decisions[task_id].get()


store = TaskStore()
