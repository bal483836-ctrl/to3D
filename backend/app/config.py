"""集中式配置：全部通过环境变量注入，便于容器化与多环境部署。"""
from __future__ import annotations

import os


class Settings:
    # 适配器：mock（默认，无 GPU 可跑）或 http（接真实混元 3D 服务）
    adapter: str = os.getenv("TO3D_ADAPTER", "mock")
    # 当 adapter=http 时必填：混元 3D 推理服务地址
    hunyuan_endpoint: str = os.getenv("TO3D_HUNYUAN_ENDPOINT", "")
    # 产物（glb/obj）输出目录
    output_dir: str = os.getenv("TO3D_OUTPUT_DIR", "/tmp/to3d_outputs")
    # 推理请求超时（秒）
    request_timeout: float = float(os.getenv("TO3D_REQUEST_TIMEOUT", "600"))


settings = Settings()
