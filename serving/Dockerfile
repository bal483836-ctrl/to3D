# 混元 3D 桥接推理服务（GPU / CUDA）
# 需要 NVIDIA 驱动 + nvidia-container-toolkit，运行：
#   docker build -f serving/Dockerfile -t to3d-hunyuan:latest serving
#   docker run --gpus all -p 9000:9000 to3d-hunyuan:latest
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PYOPENGL_PLATFORM=egl HF_HOME=/data/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip git libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch（CUDA 12.1）
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 混元 hy3dgen（源码安装）+ 桥接依赖
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /opt/Hunyuan3D-2 \
    && cd /opt/Hunyuan3D-2 \
    && pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir -e . \
    && pip3 install --no-cache-dir -e hy3dgen/texgen/custom_rasterizer \
    && pip3 install --no-cache-dir -e hy3dgen/texgen/differentiable_renderer

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir fastapi "uvicorn[standard]" trimesh pillow httpx numpy pyrender cn_clip

COPY hunyuan_server.py /app/hunyuan_server.py

EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:9000/health')" || exit 1
CMD ["python3", "hunyuan_server.py"]
