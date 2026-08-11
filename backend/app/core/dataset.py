"""后续任务：对已生成的模型渲染 50 视角四模态数据集。

复用 dataset_tool 的 Blender 脚本与参考相机参数（不重复实现），仅在此编排：
定位 Blender → 跑子进程 → 打包 zip。纯定位/查询函数可脱离 Blender 单测。
"""
from __future__ import annotations

import asyncio
import os
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings
from app.observability import logger

# 复用独立工具的脚本与参考 meta（仓库根/dataset_tool）
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TOOL = os.path.join(_REPO_ROOT, "dataset_tool")
BLENDER_SCRIPT = os.path.join(_TOOL, "blender_render_views.py")
REFERENCE_META = os.path.join(_TOOL, "reference_meta_data.json")


@dataclass
class DatasetJob:
    dataset_id: str
    status: str = "queued"  # queued|running|done|failed
    progress: float = 0.0
    zip_path: Optional[str] = None
    error: Optional[str] = None
    log_tail: list[str] = field(default_factory=list)


_jobs: dict[str, DatasetJob] = {}


def get_job(dataset_id: str) -> Optional[DatasetJob]:
    return _jobs.get(dataset_id)


def new_dataset_id() -> str:
    return "ds_" + uuid.uuid4().hex[:12]


def find_blender() -> Optional[str]:
    cand = settings.blender_bin
    if os.path.isabs(cand) and os.path.isfile(cand):
        return cand
    return shutil.which(cand)


def _zip_dir(src: str, dst: str) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src):
            for fn in files:
                fp = os.path.join(root, fn)
                zf.write(fp, os.path.relpath(fp, src))


async def run_dataset_job(dataset_id: str, model_path: str) -> None:
    job = _jobs[dataset_id]
    try:
        blender = find_blender()
        if blender is None:
            raise RuntimeError(
                f"未找到 Blender（TO3D_BLENDER_BIN={settings.blender_bin}）。"
                "请安装 Blender 3.x/4.x 并设置 TO3D_BLENDER_BIN 指向可执行文件。"
            )
        if not os.path.isfile(model_path):
            raise RuntimeError(f"模型文件不存在: {model_path}")

        out_dir = os.path.join(settings.dataset_dir, dataset_id)
        os.makedirs(out_dir, exist_ok=True)
        job.status = "running"
        cmd = [
            blender, "-b", "--python", BLENDER_SCRIPT, "--",
            "--model", model_path, "--meta", REFERENCE_META, "--out", out_dir,
            "--engine", settings.dataset_engine, "--samples", str(settings.dataset_samples),
        ]
        logger.info("后续任务 %s 启动: %s", dataset_id, " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="ignore").rstrip()
            job.log_tail = (job.log_tail + [line])[-40:]
            if "[render] view" in line:
                try:
                    job.progress = min(0.95, int(line.split("view", 1)[1].split("/", 1)[0]) / 50)
                except ValueError:
                    pass
        rc = await asyncio.wait_for(proc.wait(), timeout=settings.dataset_timeout)
        if rc != 0:
            raise RuntimeError(f"Blender 退出码 {rc}；日志：{job.log_tail[-5:]}")

        zip_path = os.path.join(settings.dataset_dir, f"{dataset_id}.zip")
        _zip_dir(out_dir, zip_path)
        job.zip_path = zip_path
        job.progress = 1.0
        job.status = "done"
        logger.info("后续任务 %s 完成 → %s", dataset_id, zip_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("后续任务失败 %s", dataset_id)
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
