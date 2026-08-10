"""FastAPI 应用：生成任务 API + WebSocket 进度 + 人在环决策（方案 §7）。

生产要素：鉴权、CORS 白名单、请求大小限制、图片来源校验(防 SSRF/LFI)、
并发上限、结构化日志 + request-id、/ready 就绪检查、/metrics 指标、
产物 TTL 清理、优雅关闭。
"""
from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.cleanup import cleanup_loop
from app.config import settings
from app.core import orchestrator
from app.models.schemas import DecisionRequest, GenerationRequest, TaskState, TaskStatus
from app.observability import (
    HTTP_REQUESTS,
    logger,
    metrics_response,
    request_id_var,
    setup_logging,
)
from app.security import ImageSourceError, check_ws_token, require_api_key, validate_images
from app.store import store

# 保持后台任务的强引用，防止被 GC 中途回收
_background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("启动：adapter=%s db=%s 并发上限=%d",
                settings.adapter, settings.db_url or "memory", settings.max_concurrency)
    stop = asyncio.Event()
    cleaner = asyncio.create_task(cleanup_loop(stop))
    try:
        yield
    finally:
        stop.set()
        for t in list(_background_tasks):
            t.cancel()
        await asyncio.gather(cleaner, *_background_tasks, return_exceptions=True)
        logger.info("已优雅关闭")


app = FastAPI(title="图文联合 3D 生成服务", version="1.0.0", lifespan=lifespan)

# CORS：默认仅同源（前端由本服务托管，无需 CORS）；配置了才开放
if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def _observe_and_limit(request: Request, call_next):
    request_id_var.set(uuid.uuid4().hex[:12])
    # 请求体大小限制
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > settings.max_request_bytes:
        return JSONResponse({"detail": "请求体过大"}, status_code=413)
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - 统一异常兜底，避免泄漏堆栈
        logger.exception("未处理异常 %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "内部错误"}, status_code=500)
    HTTP_REQUESTS.labels(
        method=request.method, path=request.url.path, code=response.status_code
    ).inc()
    return response


# --- 健康 / 就绪 / 指标 -------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    """存活探针：进程在跑即 200。"""
    return {"status": "ok", "adapter": settings.adapter}


@app.get("/api/ready")
async def ready():
    """就绪探针：依赖就位才 200（k8s readinessProbe 用）。"""
    if settings.adapter == "http" and not settings.hunyuan_endpoint:
        return JSONResponse({"status": "not_ready", "reason": "缺少 TO3D_HUNYUAN_ENDPOINT"}, 503)
    return {"status": "ready", "adapter": settings.adapter}


@app.get("/metrics")
async def metrics():
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


# --- 生成 API（鉴权保护）-----------------------------------------------------
@app.post("/api/v1/generation", dependencies=[Depends(require_api_key)])
async def create_generation(req: GenerationRequest) -> dict:
    try:
        validate_images(req.images)  # 防 SSRF / LFI / 超大图
    except ImageSourceError as e:
        raise HTTPException(400, f"图片来源不合法：{e}")

    task_id = orchestrator.new_task_id()
    state = TaskState(task_id=task_id, status=TaskStatus.queued)
    store.create(state)
    task = asyncio.create_task(orchestrator.run_generation(task_id, req))
    _background_tasks.add(task)

    def _done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        if not t.cancelled() and t.exception():
            logger.error("任务 %s 异常已记录", task_id)

    task.add_done_callback(_done)
    logger.info("创建任务 %s", task_id)
    return {"task_id": task_id, "status": state.status.value, "ws": f"/ws/tasks/{task_id}"}


@app.get("/api/v1/generation/{task_id}", dependencies=[Depends(require_api_key)])
async def get_generation(task_id: str) -> TaskState:
    state = store.get(task_id)
    if state is None:
        raise HTTPException(404, "任务不存在")
    return state


@app.post("/api/v1/generation/{task_id}/decision", dependencies=[Depends(require_api_key)])
async def post_decision(task_id: str, decision: DecisionRequest) -> dict:
    state = store.get(task_id)
    if state is None:
        raise HTTPException(404, "任务不存在")
    if state.status != TaskStatus.awaiting_user:
        raise HTTPException(409, "当前任务不在等待用户决策状态")
    await store.push_decision(task_id, decision)
    return {"ok": True}


@app.get("/api/v1/generation/{task_id}/mesh", dependencies=[Depends(require_api_key)])
async def get_mesh(task_id: str):
    state = store.get(task_id)
    if state is None or state.outputs is None or not state.outputs.glb:
        raise HTTPException(404, "模型尚未就绪")
    if not os.path.isfile(state.outputs.glb):
        raise HTTPException(410, "产物已过期清理")
    return FileResponse(
        state.outputs.glb, media_type="model/gltf-binary", filename=f"{task_id}.glb"
    )


@app.websocket("/ws/tasks/{task_id}")
async def ws_task(websocket: WebSocket, task_id: str) -> None:
    # WS 鉴权：?token=<api_key>（配置了 API Key 时强制）
    if not check_ws_token(websocket.query_params.get("token")):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    state = store.get(task_id)
    if state is None:
        await websocket.close(code=4404)
        return
    q = store.subscribe(task_id)
    try:
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


# 静态前端（存在时挂载，放在最后以免拦截 API/WS）
_frontend = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
