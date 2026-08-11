"""启动时自动加载仓库根目录的 `.env`。

为什么需要它：`.env.example` 一直写着"复制为 .env"，但没有任何地方读它——
`make dev` 从 `backend/` 启动，进程环境里压根没有 `TO3D_ADAPTER=tencent`，
于是无论 `.env` 怎么填，服务都静默地跑在 mock 上（日志里 `adapter=mock`），
表现为"能跑但只会生成通用花瓶"。

约定：
- **真实环境变量优先**：已在进程环境里设置的键不被 `.env` 覆盖
  （12-factor；Docker/compose/CI 注入的值必须压过文件）。
- 查找顺序：`TO3D_ENV_FILE` 指定的路径 → 仓库根 `.env`。
- 解析不引入第三方依赖，但覆盖 `.env.example` 的真实写法：`export` 前缀、
  引号、以及 `KEY=GLB   # 注释` 这类**行内注释**（不剥掉就会把注释读进值里）。
"""
from __future__ import annotations

import os

# backend/app/dotenv.py → 仓库根
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ENV_PATH = os.path.join(_REPO_ROOT, ".env")


def _strip_inline_comment(value: str) -> str:
    """剥掉未被引号包裹的行内注释（`#` 前需有空白，与 dotenv 一致）。

    `pass#1` 里的 `#` 是值的一部分，`GLB   # 格式` 里的不是。
    """
    out: list[str] = []
    quote: str | None = None
    for i, ch in enumerate(value):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or value[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).strip()


def parse_env(text: str) -> dict[str, str]:
    """把 .env 文本解析为键值对。非法行安静跳过，不让配置文件搞崩启动。"""
    data: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]  # 引号内原样保留，含 # 与空格
        else:
            value = _strip_inline_comment(value)
        data[key] = value
    return data


def load_dotenv(path: str | None = None, override: bool = False) -> tuple[str | None, int]:
    """加载 .env 到 os.environ，返回 (实际读取的文件, 生效的键数)。

    override=False 时，已存在于进程环境的键保持不变。
    """
    target = path or os.getenv("TO3D_ENV_FILE") or DEFAULT_ENV_PATH
    if not os.path.isfile(target):
        return None, 0
    try:
        with open(target, encoding="utf-8") as fh:
            pairs = parse_env(fh.read())
    except OSError:
        return None, 0
    applied = 0
    for key, value in pairs.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied += 1
    return target, applied
