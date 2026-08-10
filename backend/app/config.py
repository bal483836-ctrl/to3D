"""集中式配置：全部通过环境变量注入，便于容器化与多环境部署。"""
from __future__ import annotations

import os


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _list(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else []


def _json_dict(name: str) -> dict:
    import json

    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except ValueError:
        return {}


class Settings:
    # --- 适配器 / 推理 ---
    # mock(无GPU) | http(自建混元桥接) | tencent(腾讯云 ai3d API,无需 GPU)
    adapter: str = os.getenv("TO3D_ADAPTER", "mock")
    hunyuan_endpoint: str = os.getenv("TO3D_HUNYUAN_ENDPOINT", "")
    output_dir: str = os.getenv("TO3D_OUTPUT_DIR", "/tmp/to3d_outputs")
    request_timeout: float = float(os.getenv("TO3D_REQUEST_TIMEOUT", "600"))
    # 适配器调用失败重试次数（网络抖动等）
    adapter_retries: int = _int("TO3D_ADAPTER_RETRIES", 2)

    # --- 腾讯云混元生3D (ai3d) ---
    tencent_secret_id: str = os.getenv("TENCENT_SECRET_ID", "")
    tencent_secret_key: str = os.getenv("TENCENT_SECRET_KEY", "")
    tencent_region: str = os.getenv("TENCENT_REGION", "ap-guangzhou")
    tencent_ai3d_endpoint: str = os.getenv("TENCENT_AI3D_ENDPOINT", "ai3d.tencentcloudapi.com")
    tencent_ai3d_version: str = os.getenv("TENCENT_AI3D_VERSION", "2025-05-13")
    tencent_result_format: str = os.getenv("TENCENT_RESULT_FORMAT", "GLB")  # GLB|OBJ|STL|USDZ|FBX
    tencent_enable_pbr: bool = _bool("TENCENT_ENABLE_PBR", True)
    tencent_poll_interval: float = float(os.getenv("TENCENT_POLL_INTERVAL", "5"))
    tencent_poll_timeout: float = float(os.getenv("TENCENT_POLL_TIMEOUT", "600"))
    # 对象存储(COS)：多视图需公网图片 URL，本地多图会先上传到此桶再提交
    tencent_cos_bucket: str = os.getenv("TENCENT_COS_BUCKET", "")  # 形如 name-1250000000
    tencent_cos_region: str = os.getenv("TENCENT_COS_REGION", "") or os.getenv("TENCENT_REGION", "ap-guangzhou")
    # 视角映射覆盖/扩展：JSON，如 {"top":"top","bottom":"bottom","left45":"left_front"}
    # 默认仅 front/back/left/right（保证不因未知 ViewType 报错）；据你账号文档补齐 8 视角
    tencent_view_map: dict = _json_dict("TENCENT_VIEW_MAP")

    # --- 持久化 ---
    # 空 → 进程内存；sqlite:///abs/path.db → SQLite 持久化（单机生产可用）
    db_url: str = os.getenv("TO3D_DB_URL", "")

    # --- 安全 ---
    # 非空则启用鉴权：请求需带 Authorization: Bearer <key> 或 X-API-Key
    api_key: str = os.getenv("TO3D_API_KEY", "")
    # CORS 允许来源；默认空=仅同源。设 "*" 或逗号分隔域名开放跨域
    allowed_origins: list[str] = _list("TO3D_ALLOWED_ORIGINS")
    # 单张图片(data-uri 解码后)字节上限
    max_image_bytes: int = _int("TO3D_MAX_IMAGE_BYTES", 10 * 1024 * 1024)
    # 请求体字节上限（防超大 payload 打爆内存）
    max_request_bytes: int = _int("TO3D_MAX_REQUEST_BYTES", 64 * 1024 * 1024)
    # 是否禁止抓取私网/环回地址（防 SSRF），默认禁止
    block_private_ips: bool = _bool("TO3D_BLOCK_PRIVATE_IPS", True)
    # 是否允许本地文件路径作为图片来源（默认禁止，防 LFI）
    allow_local_file_images: bool = _bool("TO3D_ALLOW_LOCAL_FILE_IMAGES", False)

    # --- 并发 / 生命周期 ---
    # 同时进行的生成任务上限（GPU 通常 1~2）
    max_concurrency: int = _int("TO3D_MAX_CONCURRENCY", 2)
    # 任务与产物保留秒数，过期清理
    task_ttl_seconds: int = _int("TO3D_TASK_TTL_SECONDS", 3600)
    # 清理扫描间隔
    cleanup_interval_seconds: int = _int("TO3D_CLEANUP_INTERVAL", 600)

    # --- Blender 50 视角数据集 ---
    blender_bin: str = os.getenv("TO3D_BLENDER_BIN", "blender")  # blender 可执行文件
    dataset_dir: str = os.getenv("TO3D_DATASET_DIR", "/tmp/to3d_datasets")
    dataset_engine: str = os.getenv("TO3D_DATASET_ENGINE", "CYCLES")  # CYCLES|BLENDER_EEVEE
    dataset_samples: int = _int("TO3D_DATASET_SAMPLES", 64)
    dataset_timeout: float = float(os.getenv("TO3D_DATASET_TIMEOUT", "3600"))


settings = Settings()
