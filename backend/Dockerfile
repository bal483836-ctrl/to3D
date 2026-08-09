# 图文联合 3D 生成 · 后端(含前端静态资源)
# 构建上下文为仓库根目录：docker build -f backend/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TO3D_OUTPUT_DIR=/data/outputs

WORKDIR /app

# trimesh/pillow 运行期可能用到的最小系统库
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用层缓存
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# 应用代码 + 前端（main.py 以 ../../frontend 相对路径挂载静态资源）
COPY backend backend
COPY frontend frontend

RUN mkdir -p /data/outputs
WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
