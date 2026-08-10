"""mesh 下载在开启 API Key 时支持 ?token= 鉴权（浏览器 <a download> 场景）。"""
from __future__ import annotations

from app import security


def test_is_authorized_matrix(monkeypatch):
    # 配置了 key：头部或 token 任一命中才放行
    monkeypatch.setattr(security.settings, "api_key", "secret")
    assert security.is_authorized(token="secret") is True
    assert security.is_authorized(authorization="Bearer secret") is True
    assert security.is_authorized(x_api_key="secret") is True
    assert security.is_authorized(token="wrong") is False
    assert security.is_authorized() is False

    # 未配置 key → 一律放行
    monkeypatch.setattr(security.settings, "api_key", "")
    assert security.is_authorized() is True
