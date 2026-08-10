"""50 视角多模态数据集生成：调用 Blender 脚本，产出 Color/Depth/Normal/Mask + meta。

编排层：定位 Blender、准备输入(模型 + 参考 meta)、跑 Blender 子进程、打包 zip。
纯函数(build_artifact_meta / 定位) 可脱离 Blender 单测；真实渲染需机器装 Blender。
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

# 仓库根目录（backend/app/core/dataset.py → 上三级）
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BLENDER_SCRIPT = os.path.join(_REPO_ROOT, "tools", "blender_render_views.py")
REFERENCE_META = os.path.join(_REPO_ROOT, "datasets", "reference_meta_data.json")


@dataclass
class DatasetJob:
    dataset_id: str
    status: str = "queued"  # queued|running|done|failed
    progress: float = 0.0
    out_dir: str = ""
    zip_path: Optional[str] = None
    error: Optional[str] = None
    log_tail: list[str] = field(default_factory=list)


_jobs: dict[str, DatasetJob] = {}


def get_job(dataset_id: str) -> Optional[DatasetJob]:
    return _jobs.get(dataset_id)


def find_blender() -> Optional[str]:
    """返回可用的 blender 可执行路径，找不到返回 None。"""
    cand = settings.blender_bin
    if os.path.isabs(cand) and os.path.isfile(cand):
        return cand
    return shutil.which(cand)


def build_artifact_meta(ref: dict) -> dict:
    """按参考结构生成单文物 meta（相机参数原样，补四模态路径）。纯函数，可单测。"""
    out = {
        "camera_model": ref["camera_model"],
        "height": ref["height"],
        "width": ref["width"],
        "worldtogt": ref["worldtogt"],
        "scene_box": ref["scene_box"],
        "frames": [],
    }
    for i, f in enumerate(ref["frames"]):
        out["frames"].append({
            "rgb_path": f"{i}_colors.png",
            "depth_path": f"{i}_depth.exr",
            "normal_path": f"{i}_normal.png",
            "mask_path": f"{i}_mask.png",
            "camtoworld": f["camtoworld"],
            "intrinsics": f["intrinsics"],
        })
    return out


def new_dataset_id() -> str:
    return "ds_" + uuid.uuid4().hex[:12]


def _zip_dir(src_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                zf.write(fp, os.path.relpath(fp, src_dir))


async def run_dataset_job(dataset_id: str, model_path: str) -> None:
    """在后台跑 Blender 生成 50 视角四模态数据集并打包。"""
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
        if not os.path.isfile(REFERENCE_META):
            raise RuntimeError(f"参考 meta 缺失: {REFERENCE_META}")

        out_dir = os.path.join(settings.dataset_dir, dataset_id)
        os.makedirs(out_dir, exist_ok=True)
        job.out_dir = out_dir
        job.status = "running"

        cmd = [
            blender, "-b", "--python", BLENDER_SCRIPT, "--",
            "--model", model_path, "--meta", REFERENCE_META, "--out", out_dir,
            "--engine", settings.dataset_engine, "--samples", str(settings.dataset_samples),
        ]
        logger.info("数据集任务 %s 启动: %s", dataset_id, " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        total = 50
        async for raw in proc.stdout:
            line = raw.decode(errors="ignore").rstrip()
            job.log_tail = (job.log_tail + [line])[-40:]
            if "[render] view" in line:
                try:
                    done = int(line.split("view", 1)[1].split("/", 1)[0])
                    job.progress = min(0.95, done / total)
                except ValueError:
                    pass
        rc = await asyncio.wait_for(proc.wait(), timeout=settings.dataset_timeout)
        if rc != 0:
            raise RuntimeError(f"Blender 退出码 {rc}；日志尾部：{job.log_tail[-5:]}")

        zip_path = os.path.join(settings.dataset_dir, f"{dataset_id}.zip")
        _zip_dir(out_dir, zip_path)
        job.zip_path = zip_path
        job.progress = 1.0
        job.status = "done"
        logger.info("数据集任务 %s 完成 → %s", dataset_id, zip_path)
    except Exception as exc:  # noqa: BLE001 - 顶层兜底
        logger.exception("数据集任务失败 %s", dataset_id)
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
