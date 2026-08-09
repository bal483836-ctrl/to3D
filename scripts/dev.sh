#!/usr/bin/env bash
# 一条命令本地起服务：创建 venv、装依赖、启动(热重载)。
# 用法: ./scripts/dev.sh   访问 http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "[dev] 创建虚拟环境 .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[dev] 安装依赖"
pip install -U pip >/dev/null
pip install -r backend/requirements-dev.txt >/dev/null

cd backend
echo "[dev] 启动 http://127.0.0.1:${PORT:-8000}"
exec uvicorn app.main:app --reload --host 0.0.0.0 --port "${PORT:-8000}"
