"""独立小工具：上传 3D 模型 → 按参考 50 组相机参数渲染 Color/Depth/Normal/Mask。

与主产品解耦，专为本地/VSCode 运行。启动后浏览器上传模型即可生成数据集 zip。

运行：
    pip install -r dataset_tool/requirements.txt
    cd dataset_tool && uvicorn app:app --port 8100   # 或 python app.py
    打开 http://127.0.0.1:8100

需要机器已安装 Blender 3.x/4.x，并让 TO3D_BLENDER_BIN 指向可执行文件（默认 "blender"）。
"""
from __future__ import annotations

import asyncio
import os
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))

# 自动加载本目录 .env（配置 Blender 路径等），未装 python-dotenv 时跳过
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(HERE, ".env"))
except ImportError:
    pass
BLENDER_SCRIPT = os.path.join(HERE, "blender_render_views.py")
REFERENCE_META = os.path.join(HERE, "reference_meta_data.json")

WORKDIR = os.environ.get("TO3D_DS_WORKDIR", os.path.join(HERE, "_work"))
BLENDER_BIN = os.environ.get("TO3D_BLENDER_BIN", "blender")
ENGINE = os.environ.get("TO3D_DATASET_ENGINE", "CYCLES")
SAMPLES = os.environ.get("TO3D_DATASET_SAMPLES", "64")
TIMEOUT = float(os.environ.get("TO3D_DATASET_TIMEOUT", "3600"))
MAX_MODEL_BYTES = int(os.environ.get("TO3D_MAX_MODEL_BYTES", str(300 * 1024 * 1024)))
ALLOWED_EXT = {".glb", ".gltf", ".obj", ".fbx", ".ply"}


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued|running|done|failed
    progress: float = 0.0
    zip_path: Optional[str] = None
    error: Optional[str] = None
    log_tail: list[str] = field(default_factory=list)


jobs: dict[str, Job] = {}
app = FastAPI(title="50 视角数据集生成工具", version="1.0.0")


def find_blender() -> Optional[str]:
    if os.path.isabs(BLENDER_BIN) and os.path.isfile(BLENDER_BIN):
        return BLENDER_BIN
    return shutil.which(BLENDER_BIN)


def _zip_dir(src: str, dst: str) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src):
            for fn in files:
                fp = os.path.join(root, fn)
                zf.write(fp, os.path.relpath(fp, src))


async def _run(job_id: str, model_path: str) -> None:
    job = jobs[job_id]
    try:
        blender = find_blender()
        if blender is None:
            raise RuntimeError(
                f"未找到 Blender（TO3D_BLENDER_BIN={BLENDER_BIN}）。"
                "请安装 Blender 3.x/4.x 并设置 TO3D_BLENDER_BIN 指向可执行文件。"
            )
        out_dir = os.path.join(WORKDIR, job_id, "out")
        os.makedirs(out_dir, exist_ok=True)
        job.status = "running"
        cmd = [
            blender, "-b", "--python", BLENDER_SCRIPT, "--",
            "--model", model_path, "--meta", REFERENCE_META, "--out", out_dir,
            "--engine", ENGINE, "--samples", SAMPLES,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="ignore").rstrip()
            job.log_tail = (job.log_tail + [line])[-60:]
            if "[render] view" in line:
                try:
                    job.progress = min(0.95, int(line.split("view", 1)[1].split("/", 1)[0]) / 50)
                except ValueError:
                    pass
        rc = await asyncio.wait_for(proc.wait(), timeout=TIMEOUT)
        if rc != 0:
            raise RuntimeError(f"Blender 退出码 {rc}；日志：{job.log_tail[-5:]}")
        zip_path = os.path.join(WORKDIR, f"{job_id}.zip")
        _zip_dir(out_dir, zip_path)
        job.zip_path = zip_path
        job.progress = 1.0
        job.status = "done"
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "blender": find_blender() or "NOT FOUND"}


@app.post("/api/dataset")
async def create(model: UploadFile = File(...)) -> dict:
    ext = os.path.splitext(model.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"仅支持 {sorted(ALLOWED_EXT)}，收到 {ext or '无扩展名'}")
    data = await model.read()
    if len(data) > MAX_MODEL_BYTES:
        raise HTTPException(413, f"模型超过大小上限 {MAX_MODEL_BYTES} 字节")
    if not data:
        raise HTTPException(400, "空文件")

    job_id = "ds_" + uuid.uuid4().hex[:12]
    up_dir = os.path.join(WORKDIR, job_id)
    os.makedirs(up_dir, exist_ok=True)
    model_path = os.path.join(up_dir, f"model{ext}")
    with open(model_path, "wb") as f:
        f.write(data)

    jobs[job_id] = Job(id=job_id)
    asyncio.create_task(_run(job_id, model_path))
    return {"dataset_id": job_id, "status": "queued"}


@app.get("/api/dataset/{job_id}")
async def status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return {
        "dataset_id": job.id, "status": job.status, "progress": job.progress,
        "error": job.error, "log_tail": job.log_tail[-8:],
        "download": f"/api/dataset/{job_id}/download" if job.status == "done" else None,
    }


@app.get("/api/dataset/{job_id}/download")
async def download(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.status != "done" or not job.zip_path:
        raise HTTPException(404, "数据集尚未就绪")
    if not os.path.isfile(job.zip_path):
        raise HTTPException(410, "数据集文件缺失")
    return FileResponse(job.zip_path, media_type="application/zip", filename=f"{job_id}.zip")


# 静态上传页
app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    os.makedirs(WORKDIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8100")))
