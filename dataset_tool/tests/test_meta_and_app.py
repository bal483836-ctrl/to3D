"""meta 结构 + 上传端点（无 Blender 亦可测编排/校验）。"""
from __future__ import annotations

import io
import json
import os
import time

from fastapi.testclient import TestClient

import app as tool_app
from meta_utils import build_artifact_meta

REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "reference_meta_data.json")


def test_meta_structure_matches_reference():
    ref = json.load(open(REF, encoding="utf-8"))
    meta = build_artifact_meta(ref)
    assert set(meta) == {"camera_model", "height", "width", "worldtogt", "scene_box", "frames"}
    assert len(meta["frames"]) == 50
    f0, r0 = meta["frames"][0], ref["frames"][0]
    assert f0["camtoworld"] == r0["camtoworld"]      # 相机参数原样
    assert f0["intrinsics"] == r0["intrinsics"]
    assert (f0["rgb_path"], f0["depth_path"], f0["normal_path"], f0["mask_path"]) == \
           ("0_colors.png", "0_depth.exr", "0_normal.png", "0_mask.png")


def test_upload_rejects_bad_extension():
    with TestClient(tool_app.app) as client:
        r = client.post("/api/dataset",
                        files={"model": ("bad.txt", io.BytesIO(b"x"), "text/plain")})
        assert r.status_code == 400


def test_upload_runs_and_reports_missing_blender(monkeypatch):
    monkeypatch.setattr(tool_app, "find_blender", lambda: None)
    with TestClient(tool_app.app) as client:
        r = client.post("/api/dataset",
                        files={"model": ("m.glb", io.BytesIO(b"GLB-bytes"), "model/gltf-binary")})
        assert r.status_code == 200
        ds = r.json()["dataset_id"]
        for _ in range(40):
            s = client.get(f"/api/dataset/{ds}").json()
            if s["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert s["status"] == "failed" and "Blender" in s["error"]


def test_custom_meta_rejected_when_invalid(monkeypatch):
    monkeypatch.setattr(tool_app, "find_blender", lambda: None)
    with TestClient(tool_app.app) as client:
        r = client.post("/api/dataset", files={
            "model": ("m.glb", io.BytesIO(b"GLB"), "model/gltf-binary"),
            "meta": ("meta_data.json", io.BytesIO(b"{not json"), "application/json"),
        })
        assert r.status_code == 400


def test_custom_meta_accepted(monkeypatch):
    monkeypatch.setattr(tool_app, "find_blender", lambda: None)
    good = json.dumps({"camera_model": "OPENCV", "height": 800, "width": 800,
                       "frames": [{"rgb_path": "0_colors.png"}]}).encode()
    with TestClient(tool_app.app) as client:
        r = client.post("/api/dataset", files={
            "model": ("m.glb", io.BytesIO(b"GLB"), "model/gltf-binary"),
            "meta": ("meta_data.json", io.BytesIO(good), "application/json"),
        })
        assert r.status_code == 200  # 合法 meta 被接受（随后因无 Blender 而失败，属预期）


def test_health():
    with TestClient(tool_app.app) as client:
        assert client.get("/api/health").status_code == 200
