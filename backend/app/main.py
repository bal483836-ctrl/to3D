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

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import config
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
from app.security import (
    ImageSourceError,
    check_ws_token,
    is_authorized,
    require_api_key,
    validate_images,
)
from app.store import store

# 保持后台任务的强引用，防止被 GC 中途回收
_background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # 配置来源要显式说出来：此前 .env 从不被读取，用户填了 tencent 也静默跑 mock
    if config.ENV_FILE:
        logger.info("已加载配置文件 %s（生效 %d 项；已存在的环境变量优先）",
                    config.ENV_FILE, config.ENV_KEYS_APPLIED)
    else:
        logger.info("未找到 .env（查找路径 %s），仅使用进程环境变量。"
                    "如需接真实模型：cp .env.example .env 并填写",
                    os.getenv("TO3D_ENV_FILE") or config.DEFAULT_ENV_PATH)
    logger.info("启动：adapter=%s db=%s 并发上限=%d",
                settings.adapter, settings.db_url or "memory", settings.max_concurrency)
    if settings.adapter == "mock":
        logger.warning(
            "当前为 mock 适配器：产出的是占位网格（不读取你上传的图片内容），"
            "仅供无 GPU 时跑通链路。接真实模型请设 TO3D_ADAPTER=tencent 或 http"
            "（只填凭据不改这一项仍然是 mock）"
        )
    for warn in config.config_warnings():
        logger.warning("配置提示：%s", warn)
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


# 模型上传走单独的大小上限：3D 文件通常远大于 JSON 请求体
_MODEL_UPLOAD_PATH = "/api/v1/dataset/upload"


@app.middleware("http")
async def _observe_and_limit(request: Request, call_next):
    request_id_var.set(uuid.uuid4().hex[:12])
    # 请求体大小限制
    limit = (
        settings.max_model_bytes
        if request.url.path == _MODEL_UPLOAD_PATH
        else settings.max_request_bytes
    )
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > limit:
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
    """存活探针：进程在跑即 200。

    附带配置来源与 mock 标记，方便一眼确认「.env 到底有没有被读到」——
    不必再去翻启动日志猜为什么还在跑占位模型。
    """
    return {
        "status": "ok",
        "adapter": settings.adapter,
        "is_mock": settings.adapter == "mock",
        "env_file": config.ENV_FILE,
        "warnings": config.config_warnings(),
    }


@app.get("/api/ready")
async def ready():
    """就绪探针：依赖就位才 200（k8s readinessProbe 用）。"""
    if settings.adapter == "http" and not settings.hunyuan_endpoint:
        return JSONResponse({"status": "not_ready", "reason": "缺少 TO3D_HUNYUAN_ENDPOINT"}, 503)
    if settings.adapter == "tencent" and not (
        settings.tencent_secret_id and settings.tencent_secret_key
    ):
        return JSONResponse(
            {"status": "not_ready", "reason": "缺少 TENCENT_SECRET_ID / TENCENT_SECRET_KEY"}, 503
        )
    return {"status": "ready", "adapter": settings.adapter, "env_file": config.ENV_FILE}


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


@app.get("/api/v1/generation/{task_id}/mesh")
async def get_mesh(
    task_id: str,
    format: str = "glb",
    token: str | None = None,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """下载产物。format=glb|obj。

    浏览器 <a download> / 3D 预览无法带鉴权头，故除头部外也接受 ?token=<key>，
    与 WebSocket 一致（未配置 API Key 时二者皆放行）。
    """
    if not is_authorized(authorization, x_api_key, token):
        raise HTTPException(401, "未授权：缺少或错误的 API Key")
    state = store.get(task_id)
    if state is None or state.outputs is None:
        raise HTTPException(404, "模型尚未就绪")
    fmt = format.lower()
    if fmt not in {"glb", "obj"}:
        raise HTTPException(400, "format 仅支持 glb 或 obj")
    path = state.outputs.glb if fmt == "glb" else state.outputs.obj
    media = "model/gltf-binary" if fmt == "glb" else "text/plain"
    if not path:
        raise HTTPException(404, "该格式产物不存在")
    if not os.path.isfile(path):
        raise HTTPException(410, "产物已过期清理")
    return FileResponse(path, media_type=media, filename=f"{task_id}.{fmt}")


@app.post("/api/v1/generation/{task_id}/dataset", dependencies=[Depends(require_api_key)])
async def create_dataset(task_id: str) -> dict:
    """后续任务：对已生成的模型渲染 50 视角 × 四模态(Color/Depth/Normal/Mask) 数据集。"""
    from app.core import dataset

    state = store.get(task_id)
    if state is None or state.outputs is None or not state.outputs.glb:
        raise HTTPException(404, "模型尚未就绪，无法生成数据集")
    if not os.path.isfile(state.outputs.glb):
        raise HTTPException(410, "模型产物已过期清理")

    ds_id = dataset.new_dataset_id()
    dataset._jobs[ds_id] = dataset.DatasetJob(dataset_id=ds_id)
    t = asyncio.create_task(dataset.run_dataset_job(ds_id, state.outputs.glb))
    _background_tasks.add(t)
    t.add_done_callback(_background_tasks.discard)
    return {"dataset_id": ds_id, "status": "queued"}


@app.post(_MODEL_UPLOAD_PATH, dependencies=[Depends(require_api_key)])
async def upload_dataset(model: UploadFile = File(...)) -> dict:
    """直接上传已有 3D 模型（GLB/GLTF/OBJ/FBX/PLY）跑 50 视角数据集。

    与 `/generation/{id}/dataset` 的区别只是模型来源：那条路径用本服务刚生成的
    产物，这条用用户手上已有的模型，无需先跑一次生成。后续查询/下载共用同一组接口。
    """
    from app.core import dataset

    data = await model.read()
    ds_id = dataset.new_dataset_id()
    try:
        model_path = dataset.save_uploaded_model(ds_id, model.filename or "", data)
    except dataset.ModelUploadError as e:
        raise HTTPException(400, str(e))

    dataset._jobs[ds_id] = dataset.DatasetJob(dataset_id=ds_id)
    t = asyncio.create_task(dataset.run_dataset_job(ds_id, model_path))
    _background_tasks.add(t)
    t.add_done_callback(_background_tasks.discard)
    logger.info("上传模型创建数据集任务 %s（源文件 %s）", ds_id, model.filename)
    return {"dataset_id": ds_id, "status": "queued", "source": "upload"}


@app.get("/api/v1/dataset/{dataset_id}", dependencies=[Depends(require_api_key)])
async def get_dataset(dataset_id: str) -> dict:
    from app.core import dataset

    job = dataset.get_job(dataset_id)
    if job is None:
        raise HTTPException(404, "数据集任务不存在")
    return {
        "dataset_id": job.dataset_id, "status": job.status, "progress": job.progress,
        "error": job.error, "log_tail": job.log_tail[-8:],
        "download": f"/api/v1/dataset/{dataset_id}/download" if job.status == "done" else None,
    }


@app.get("/api/v1/dataset/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    token: str | None = None,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    from app.core import dataset

    if not is_authorized(authorization, x_api_key, token):
        raise HTTPException(401, "未授权")
    job = dataset.get_job(dataset_id)
    if job is None or job.status != "done" or not job.zip_path:
        raise HTTPException(404, "数据集尚未就绪")
    if not os.path.isfile(job.zip_path):
        raise HTTPException(410, "数据集已过期清理")
    return FileResponse(job.zip_path, media_type="application/zip",
                        filename=f"{dataset_id}.zip")


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
