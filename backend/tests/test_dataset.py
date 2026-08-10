"""数据集编排：meta 结构、Blender 缺失处理、API 端点。"""
from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient

from app.core import dataset
from app.main import app


def test_build_artifact_meta_structure():
    ref = json.load(open(dataset.REFERENCE_META, encoding="utf-8"))
    meta = dataset.build_artifact_meta(ref)
    assert set(meta.keys()) == {
        "camera_model", "height", "width", "worldtogt", "scene_box", "frames"}
    assert len(meta["frames"]) == 50
    f0 = meta["frames"][0]
    # 相机参数原样、补四模态路径
    assert f0["camtoworld"] == ref["frames"][0]["camtoworld"]
    assert f0["intrinsics"] == ref["frames"][0]["intrinsics"]
    assert f0["rgb_path"] == "0_colors.png"
    assert f0["depth_path"] == "0_depth.exr"
    assert f0["normal_path"] == "0_normal.png"
    assert f0["mask_path"] == "0_mask.png"


async def test_run_job_reports_missing_blender(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "find_blender", lambda: None)
    model = tmp_path / "m.glb"
    model.write_bytes(b"glb")
    ds_id = dataset.new_dataset_id()
    dataset._jobs[ds_id] = dataset.DatasetJob(dataset_id=ds_id)
    await dataset.run_dataset_job(ds_id, str(model))
    job = dataset.get_job(ds_id)
    assert job.status == "failed" and "Blender" in job.error


def test_dataset_endpoint_requires_ready_model():
    with TestClient(app) as client:
        r = client.post("/api/v1/generation/gen_nope/dataset")
        assert r.status_code == 404
        r2 = client.get("/api/v1/dataset/ds_nope")
        assert r2.status_code == 404


def test_dataset_script_and_reference_present():
    assert os.path.isfile(dataset.BLENDER_SCRIPT)
    assert os.path.isfile(dataset.REFERENCE_META)
