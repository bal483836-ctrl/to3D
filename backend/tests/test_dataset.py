"""后续任务(50 视角数据集)：定位、Blender 缺失处理、API 端点。"""
from __future__ import annotations

import os

from fastapi.testclient import TestClient

from app.core import dataset
from app.main import app


def test_reuses_dataset_tool_assets():
    # 复用 dataset_tool 的脚本与参考 meta（不重复实现）
    assert dataset.BLENDER_SCRIPT.endswith(os.path.join("dataset_tool", "blender_render_views.py"))
    assert os.path.isfile(dataset.BLENDER_SCRIPT)
    assert os.path.isfile(dataset.REFERENCE_META)


async def test_run_job_reports_missing_blender(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "find_blender", lambda: None)
    model = tmp_path / "m.glb"
    model.write_bytes(b"glb")
    ds_id = dataset.new_dataset_id()
    dataset._jobs[ds_id] = dataset.DatasetJob(dataset_id=ds_id)
    await dataset.run_dataset_job(ds_id, str(model))
    job = dataset.get_job(ds_id)
    assert job.status == "failed" and "Blender" in job.error


def test_dataset_endpoints_404():
    with TestClient(app) as client:
        assert client.post("/api/v1/generation/gen_nope/dataset").status_code == 404
        assert client.get("/api/v1/dataset/ds_nope").status_code == 404
