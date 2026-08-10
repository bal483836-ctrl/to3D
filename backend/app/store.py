"""任务存储：进程内存 或 SQLite 持久化，接口一致，按配置选择。

- 进度广播(subscribe/wait)与人在环决策为**进程内** asyncio 通道，适合单进程部署。
- SQLite 提供任务记录/结果的**持久化**（重启不丢已完成结果）。
- 多副本水平扩展需外部 Postgres(状态)+Redis(pub/sub)+任务队列，见 docs/生产部署.md。
"""
from __future__ import annotations

import abc
import asyncio
import json
import sqlite3
import threading
import time
from typing import Optional

from app.config import settings
from app.models.schemas import TaskState


class _PubSub:
    """进程内的进度广播与决策通道（两种存储实现共用）。"""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._decisions: dict[str, asyncio.Queue] = {}

    def _init_channels(self, task_id: str) -> None:
        self._subscribers.setdefault(task_id, [])
        self._decisions.setdefault(task_id, asyncio.Queue())

    async def _broadcast(self, state: TaskState) -> None:
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

    async def push_decision(self, task_id: str, decision) -> None:
        self._decisions.setdefault(task_id, asyncio.Queue())
        await self._decisions[task_id].put(decision)

    async def wait_decision(self, task_id: str):
        self._decisions.setdefault(task_id, asyncio.Queue())
        return await self._decisions[task_id].get()


class BaseTaskStore(_PubSub, abc.ABC):
    @abc.abstractmethod
    def create(self, state: TaskState) -> None: ...

    @abc.abstractmethod
    def get(self, task_id: str) -> Optional[TaskState]: ...

    @abc.abstractmethod
    async def update(self, state: TaskState) -> None: ...

    @abc.abstractmethod
    def expired(self, ttl_seconds: int) -> list[TaskState]:
        """返回已超过 TTL 的任务（用于清理产物与记录）。"""

    @abc.abstractmethod
    def delete(self, task_id: str) -> None: ...


class InMemoryTaskStore(BaseTaskStore):
    def __init__(self) -> None:
        super().__init__()
        self._tasks: dict[str, TaskState] = {}
        self._created: dict[str, float] = {}

    def create(self, state: TaskState) -> None:
        self._tasks[state.task_id] = state
        self._created[state.task_id] = time.time()
        self._init_channels(state.task_id)

    def get(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    async def update(self, state: TaskState) -> None:
        self._tasks[state.task_id] = state
        await self._broadcast(state)

    def expired(self, ttl_seconds: int) -> list[TaskState]:
        now = time.time()
        return [
            self._tasks[tid]
            for tid, ts in list(self._created.items())
            if now - ts > ttl_seconds and tid in self._tasks
        ]

    def delete(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._created.pop(task_id, None)
        self._subscribers.pop(task_id, None)
        self._decisions.pop(task_id, None)


class SqliteTaskStore(BaseTaskStore):
    """单机持久化。pub/sub 与决策仍在进程内。"""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks ("
            "task_id TEXT PRIMARY KEY, updated_at REAL, data TEXT)"
        )
        self._conn.commit()

    def create(self, state: TaskState) -> None:
        self._write(state)
        self._init_channels(state.task_id)

    def _write(self, state: TaskState) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks(task_id, updated_at, data) VALUES(?,?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET updated_at=excluded.updated_at, "
                "data=excluded.data",
                (state.task_id, time.time(), state.model_dump_json()),
            )
            self._conn.commit()

    def get(self, task_id: str) -> Optional[TaskState]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return TaskState.model_validate_json(row[0]) if row else None

    async def update(self, state: TaskState) -> None:
        self._write(state)
        await self._broadcast(state)

    def expired(self, ttl_seconds: int) -> list[TaskState]:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM tasks WHERE updated_at < ?", (cutoff,)
            ).fetchall()
        return [TaskState.model_validate_json(r[0]) for r in rows]

    def delete(self, task_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))
            self._conn.commit()
        self._subscribers.pop(task_id, None)
        self._decisions.pop(task_id, None)


def _build_store() -> BaseTaskStore:
    url = settings.db_url
    if url.startswith("sqlite:///"):
        return SqliteTaskStore(url[len("sqlite:///"):])
    return InMemoryTaskStore()


store: BaseTaskStore = _build_store()
