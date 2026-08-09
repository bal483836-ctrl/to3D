"""FastAPI 应用：生成任务 API + WebSocket 进度 + 人在环决策（方案 §7）。"""
from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core import orchestrator
from app.models.schemas import (
    DecisionRequest,
    GenerationRequest,
    TaskState,
    TaskStatus,
)
from app.store import store

app = FastAPI(title="图文双通道 3D 生成服务", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    from app.config import settings

    return {"status": "ok", "adapter": settings.adapter}


# 保持后台任务的强引用，防止被 GC 中途回收
_background_tasks: set[asyncio.Task] = set()


@app.post("/api/v1/generation")
async def create_generation(req: GenerationRequest) -> dict:
    task_id = orchestrator.new_task_id()
    state = TaskState(task_id=task_id, status=TaskStatus.queued)
    store.create(state)
    # 后台执行，立即返回
    task = asyncio.create_task(orchestrator.run_generation(task_id, req))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"task_id": task_id, "status": state.status.value, "ws": f"/ws/tasks/{task_id}"}


@app.get("/api/v1/generation/{task_id}")
async def get_generation(task_id: str) -> TaskState:
    state = store.get(task_id)
    if state is None:
        raise HTTPException(404, "任务不存在")
    return state


@app.post("/api/v1/generation/{task_id}/decision")
async def post_decision(task_id: str, decision: DecisionRequest) -> dict:
    state = store.get(task_id)
    if state is None:
        raise HTTPException(404, "任务不存在")
    if state.status != TaskStatus.awaiting_user:
        raise HTTPException(409, "当前任务不在等待用户决策状态")
    await store.push_decision(task_id, decision)
    return {"ok": True}


@app.get("/api/v1/generation/{task_id}/mesh")
async def get_mesh(task_id: str):
    state = store.get(task_id)
    if state is None or state.outputs is None or not state.outputs.glb:
        raise HTTPException(404, "模型尚未就绪")
    return FileResponse(
        state.outputs.glb, media_type="model/gltf-binary", filename=f"{task_id}.glb"
    )


@app.websocket("/ws/tasks/{task_id}")
async def ws_task(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    state = store.get(task_id)
    if state is None:
        await websocket.close(code=4404)
        return
    q = store.subscribe(task_id)
    try:
        # 先推当前状态
        await websocket.send_json(state.model_dump(mode="json"))
        if state.status in (TaskStatus.done, TaskStatus.failed):
            return
        while True:
            update: TaskState = await q.get()
            await websocket.send_json(update.model_dump(mode="json"))
            if update.status in (TaskStatus.done, TaskStatus.failed):
                break
    except WebSocketDisconnect:
        pass
    finally:
        store.unsubscribe(task_id, q)


# 静态前端（存在时挂载）
_frontend = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
