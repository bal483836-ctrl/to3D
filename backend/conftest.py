"""测试期配置隔离。

`app.config` 现在会自动加载仓库根的 `.env`——这正是我们让用户做的事
（`cp .env.example .env`）。但测试必须与本机配置无关：否则开发者一填上
`TO3D_API_KEY` 或 `TO3D_ADAPTER=tencent`，测试就会因 401 或试图连真实云端而失败。

这里在**导入 app.config 之前**把 env 文件指向一个不存在的路径，让测试始终跑在
默认配置上。需要验证加载行为的用例自己 monkeypatch `TO3D_ENV_FILE`（见
tests/test_dotenv.py），不受影响。
"""
from __future__ import annotations

import os

os.environ["TO3D_ENV_FILE"] = os.path.join(os.path.dirname(__file__), ".env.pytest-absent")
