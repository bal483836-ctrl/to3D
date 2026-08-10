"""安全相关：图片来源校验（防 SSRF / LFI）、鉴权依赖。"""
from __future__ import annotations

import base64
import ipaddress
import socket
from urllib.parse import urlparse

from fastapi import Header, HTTPException

from app.config import settings


class ImageSourceError(ValueError):
    """图片来源不合法（scheme/大小/SSRF/LFI）。"""


def _decoded_len_of_data_uri(url: str) -> int:
    # data:[<mediatype>][;base64],<data>
    try:
        header, data = url.split(",", 1)
    except ValueError as e:
        raise ImageSourceError("data URI 格式不合法") from e
    if ";base64" in header:
        # base64 长度 → 解码后近似字节数
        return (len(data) * 3) // 4
    return len(data.encode("utf-8"))


def _is_public_host(host: str) -> bool:
    """解析主机名的所有 IP，若任一为私网/环回/链路本地/保留则判为不安全。"""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False  # 解析不了，保守拒绝
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            return False
    return True


def validate_image_url(url: str) -> None:
    """校验单个图片来源。不合法则抛 ImageSourceError。

    - data:image/... → 检查大小上限。
    - http(s):// → 校验主机非私网（防 SSRF），除非显式关闭。
    - file:// 或本地路径 → 默认禁止（防 LFI）。
    """
    if not isinstance(url, str) or not url:
        raise ImageSourceError("图片 url 不能为空")

    if url.startswith("data:"):
        if not url.startswith("data:image/"):
            raise ImageSourceError("data URI 必须是 image 类型")
        if _decoded_len_of_data_uri(url) > settings.max_image_bytes:
            raise ImageSourceError(
                f"图片超过大小上限 {settings.max_image_bytes} 字节"
            )
        return

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme in {"http", "https"}:
        if not parsed.hostname:
            raise ImageSourceError("url 缺少主机名")
        if settings.block_private_ips and not _is_public_host(parsed.hostname):
            raise ImageSourceError("拒绝抓取私网/内网地址（SSRF 防护）")
        return

    if scheme == "file" or scheme == "":
        if settings.allow_local_file_images:
            return
        raise ImageSourceError("默认禁止本地文件作为图片来源（防 LFI）")

    raise ImageSourceError(f"不支持的 url scheme: {scheme}")


def validate_images(images) -> None:
    for img in images:
        url = getattr(img, "url", None)
        validate_image_url(url)


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """FastAPI 依赖：当配置了 TO3D_API_KEY 时强制鉴权。

    接受 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`。
    """
    if not is_authorized(authorization, x_api_key):
        raise HTTPException(status_code=401, detail="未授权：缺少或错误的 API Key")


def _extract_header_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    if x_api_key:
        return x_api_key.strip()
    return None


def is_authorized(
    authorization: str | None = None,
    x_api_key: str | None = None,
    token: str | None = None,
) -> bool:
    """统一鉴权判定：未配置 API Key 则放行；否则头部或 query token 任一匹配即可。"""
    if not settings.api_key:
        return True
    key = _extract_header_key(authorization, x_api_key) or token
    return key == settings.api_key


def check_ws_token(token: str | None) -> bool:
    """WebSocket 鉴权：未配置则放行；配置了则校验 query token。"""
    return is_authorized(token=token)
