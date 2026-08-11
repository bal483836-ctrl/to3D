"""后续任务(50 视角数据集)：定位、Blender 缺失处理、API 端点。"""
from __future__ import annotations

import os
import time

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


# --- 上传已有模型直接跑数据集 -------------------------------------------------


def test_save_uploaded_model_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset.settings, "dataset_dir", str(tmp_path))
    path = dataset.save_uploaded_model("ds_x", "我的陶器.glb", b"GLBDATA")
    assert os.path.isfile(path)
    assert open(path, "rb").read() == b"GLBDATA"
    assert path.endswith(os.path.join("ds_x", "input", "model.glb"))


def test_save_uploaded_model_ignores_client_filename(tmp_path, monkeypatch):
    """客户端文件名不可信：只取扩展名，绝不能按它拼路径写穿目录。"""
    monkeypatch.setattr(dataset.settings, "dataset_dir", str(tmp_path))
    path = dataset.save_uploaded_model("ds_y", "../../../../etc/passwd.glb", b"x")
    assert os.path.realpath(path).startswith(os.path.realpath(str(tmp_path)))
    assert os.path.basename(path) == "model.glb"


def test_save_uploaded_model_rejects_bad_input(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(dataset.settings, "dataset_dir", str(tmp_path))
    with pytest.raises(dataset.ModelUploadError):
        dataset.save_uploaded_model("ds_a", "x.exe", b"data")   # 格式不支持
    with pytest.raises(dataset.ModelUploadError):
        dataset.save_uploaded_model("ds_b", "x.glb", b"")       # 空文件
    with pytest.raises(dataset.ModelUploadError):
        dataset.save_uploaded_model("ds_c", "noext", b"data")   # 无扩展名

    monkeypatch.setattr(dataset.settings, "max_model_bytes", 4)
    with pytest.raises(dataset.ModelUploadError):
        dataset.save_uploaded_model("ds_d", "x.glb", b"toolong")  # 超上限


def test_upload_endpoint_accepts_glb_and_starts_job(tmp_path, monkeypatch):
    """上传 GLB → 建任务 → 可查状态。此环境无 Blender，任务应明确失败而非静默。"""
    monkeypatch.setattr(dataset.settings, "dataset_dir", str(tmp_path))
    monkeypatch.setattr(dataset, "find_blender", lambda: None)
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/dataset/upload",
            files={"model": ("陶器.glb", b"GLBDATA", "model/gltf-binary")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "queued" and body["source"] == "upload"
        ds_id = body["dataset_id"]

        for _ in range(40):
            state = client.get(f"/api/v1/dataset/{ds_id}").json()
            if state["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert state["status"] == "failed"
        assert "Blender" in state["error"]


def test_upload_endpoint_rejects_unsupported_format():
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/dataset/upload",
            files={"model": ("bad.exe", b"MZ", "application/octet-stream")},
        )
        assert r.status_code == 400
        assert ".glb" in r.json()["detail"]


def test_upload_endpoint_has_its_own_size_limit():
    """3D 文件远大于 JSON 请求体，上传路径不该被 max_request_bytes 卡住。"""
    from app.config import settings as app_settings

    assert app_settings.max_model_bytes > app_settings.max_request_bytes
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/dataset/upload",
            files={"model": ("big.glb", b"x", "model/gltf-binary")},
            headers={"Content-Length": str(app_settings.max_model_bytes + 1)},
        )
        assert r.status_code == 413
