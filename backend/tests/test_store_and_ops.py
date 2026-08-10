"""持久化存储、清理、就绪/指标端点测试。"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ModelOutputs, TaskState, TaskStatus
from app.store import InMemoryTaskStore, SqliteTaskStore


def test_sqlite_store_persists(tmp_path):
    db = str(tmp_path / "t.db")
    s1 = SqliteTaskStore(db)
    st = TaskState(task_id="gen_x", status=TaskStatus.done, progress=1.0)
    s1.create(st)
    # 新建实例(模拟重启)仍能读到
    s2 = SqliteTaskStore(db)
    got = s2.get("gen_x")
    assert got is not None and got.status == TaskStatus.done


def test_inmemory_expired_and_delete():
    s = InMemoryTaskStore()
    s.create(TaskState(task_id="gen_a", status=TaskStatus.done))
    assert s.expired(ttl_seconds=-1)  # 立刻过期
    s.delete("gen_a")
    assert s.get("gen_a") is None


def test_cleanup_removes_outputs(tmp_path, monkeypatch):
    from app import cleanup
    from app.store import store

    f = tmp_path / "gen_c_final.glb"
    f.write_bytes(b"glb")
    st = TaskState(
        task_id="gen_c", status=TaskStatus.done,
        outputs=ModelOutputs(glb=str(f)),
    )
    store.create(st)
    monkeypatch.setattr(cleanup.settings, "task_ttl_seconds", -1)  # 全部视为过期
    n = cleanup.sweep_once()
    assert n >= 1
    assert not f.exists()
    assert store.get("gen_c") is None


def test_ready_and_metrics_endpoints():
    with TestClient(app) as client:
        r = client.get("/api/ready")
        assert r.status_code == 200 and r.json()["status"] == "ready"
        m = client.get("/metrics")
        assert m.status_code == 200
