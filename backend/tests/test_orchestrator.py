"""端到端编排测试：修正闭环应收敛，API 应可用。"""
from __future__ import annotations

import time

# 1x1 PNG 的 data URI，作为合法图片来源（通过安全校验）
PX = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0"
    "lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

import pytest
from fastapi.testclient import TestClient

from app.core import orchestrator
from app.main import app
from app.models.schemas import (
    GenerationRequest,
    ImageInput,
    TaskState,
    TaskStatus,
    ViewName,
)
from app.store import store


def _make_request(**overrides):
    data = dict(
        images=[
            ImageInput(view=ViewName.front, url="x", required=True),
            ImageInput(view=ViewName.left45, url="x"),
            ImageInput(view=ViewName.right, url="x"),
        ],
        prompt="青花瓷花瓶，细长颈，圈足，缠枝莲纹，釉面高光",
    )
    data.update(overrides)
    return GenerationRequest(**data)


@pytest.mark.asyncio
async def test_correction_loop_converges():
    req = _make_request()
    task_id = orchestrator.new_task_id()
    store.create(TaskState(task_id=task_id, status=TaskStatus.queued))
    await orchestrator.run_generation(task_id, req)

    state = store.get(task_id)
    assert state.status == TaskStatus.done
    assert state.outputs is not None
    report = state.outputs.final_diff_report
    # 修正后应无残留「以文校正」动作 → 一致
    assert report.actions == []
    assert report.passed
    # 至少经历了一轮修正（初始有缺陷）
    assert state.round >= 1


@pytest.mark.asyncio
async def test_bottom_from_image_when_bottom_view_provided():
    """上传底图时，联合生成的底部直接由图像还原，无需以文修正底部。"""
    req = _make_request(
        images=[
            ImageInput(view=ViewName.front, url="x", required=True),
            ImageInput(view=ViewName.back, url="x"),
            ImageInput(view=ViewName.left, url="x"),
            ImageInput(view=ViewName.right, url="x"),
            ImageInput(view=ViewName.bottom, url="x"),
        ],
    )
    task_id = orchestrator.new_task_id()
    store.create(TaskState(task_id=task_id, status=TaskStatus.queued))
    await orchestrator.run_generation(task_id, req)
    state = store.get(task_id)
    assert state.status == TaskStatus.done
    # 有底图 → 底部不需以文修正；仍可能因花纹/材质需要一轮重绘
    assert "refine_bottom" not in state.outputs.final_diff_report.actions


def test_validation_requires_front_view():
    with pytest.raises(Exception):
        _make_request(
            images=[
                ImageInput(view=ViewName.left, url="x"),
                ImageInput(view=ViewName.right, url="x"),
            ]
        )


def test_api_create_and_fetch():
    # 用作上下文管理器以保证 portal 事件循环驱动后台任务
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/generation",
            json={
                "images": [
                    {"view": "front", "url": PX, "required": True},
                    {"view": "left45", "url": PX},
                ],
                "prompt": "陶瓷花瓶，圈足，缠枝莲纹，釉面高光",
            },
        )
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        for _ in range(80):
            state = client.get(f"/api/v1/generation/{task_id}").json()
            if state["status"] in ("done", "failed"):
                break
            time.sleep(0.25)
        assert state["status"] == "done", state
        # GLB 与 OBJ 均可下载
        glb = client.get(f"/api/v1/generation/{task_id}/mesh")
        assert glb.status_code == 200 and len(glb.content) > 0
        obj = client.get(f"/api/v1/generation/{task_id}/mesh?format=obj")
        assert obj.status_code == 200 and len(obj.content) > 0
