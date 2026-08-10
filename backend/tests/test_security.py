"""安全校验测试：图片来源(SSRF/LFI/大小) 与 API Key 鉴权。"""
from __future__ import annotations

import pytest

from app import security
from app.security import ImageSourceError, validate_image_url

PX = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def test_data_uri_ok():
    validate_image_url(PX)  # 不抛异常即通过


def test_reject_non_image_data_uri():
    with pytest.raises(ImageSourceError):
        validate_image_url("data:text/html;base64,PGh0bWw+")


def test_reject_file_scheme():
    with pytest.raises(ImageSourceError):
        validate_image_url("file:///etc/passwd")


def test_reject_bare_path_lfi():
    with pytest.raises(ImageSourceError):
        validate_image_url("/etc/passwd")


def test_reject_private_ip_ssrf():
    with pytest.raises(ImageSourceError):
        validate_image_url("http://127.0.0.1/x.png")
    with pytest.raises(ImageSourceError):
        validate_image_url("http://169.254.169.254/latest/meta-data")  # 云元数据


def test_reject_oversize_data_uri(monkeypatch):
    monkeypatch.setattr(security.settings, "max_image_bytes", 10)
    with pytest.raises(ImageSourceError):
        validate_image_url(PX)


def test_api_key_enforced(monkeypatch):
    # 打开鉴权后，未带 key 的请求应 401（monkeypatch 直接改 settings，避免 reload 污染）
    from fastapi.testclient import TestClient

    from app.main import app
    monkeypatch.setattr(security.settings, "api_key", "secret")

    with TestClient(app) as client:
        r = client.post("/api/v1/generation", json={
            "images": [{"view": "front", "url": PX, "required": True}, {"view": "left", "url": PX}],
            "prompt": "陶瓷花瓶",
        })
        assert r.status_code == 401
        assert client.get("/api/health").status_code == 200  # 健康检查不需鉴权
        # 带正确 key 应放行（进入业务，返回 200 任务已创建）
        ok = client.post("/api/v1/generation",
                         headers={"X-API-Key": "secret"},
                         json={"images": [{"view": "front", "url": PX, "required": True},
                                          {"view": "left", "url": PX}],
                               "prompt": "陶瓷花瓶"})
        assert ok.status_code == 200
