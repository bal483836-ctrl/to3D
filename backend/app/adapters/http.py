"""真实混元 3D 服务的 HTTP 适配器骨架（方案 §8）。

接入方式：部署 Hunyuan3D 推理服务（DiT + MV + Paint），暴露如下约定接口，
然后设置环境变量：

    TO3D_ADAPTER=http
    TO3D_HUNYUAN_ENDPOINT=https://your-hunyuan3d-service

约定的服务端点（可按实际服务调整映射）：
    POST /generate    {images, prompt, image_scale, text_scale}
                      -> {mesh_url/glb_base64, texture:{...}}   # 图文联合条件生成
    POST /reference   {prompt}                 -> 同上           # 规格重建，仅自检用
    POST /repaint     {mesh, prompt, focus}    -> {texture:{...}}
    POST /refine      {mesh, focus, prompt}    -> {mesh_url/glb_base64}

真实服务实现要点：把多视图图像编码(DINO/CLIP)与文字编码(CLIP text)拼接为
同一条件序列送入 Hunyuan3D-DiT，用 image_scale/text_scale 做 CFG 双模态引导，
单次去噪得到一份几何；纹理由 Hunyuan3D-Paint 联合图文条件生成。

本文件默认不被 Mock 流程加载；仅当 TO3D_ADAPTER=http 时启用。
"""
from __future__ import annotations

import base64
import io

import httpx
import trimesh

from app.adapters.base import GenerationResult, Hunyuan3DAdapter, TextureDescriptor
from app.core.preprocess import TextConstraints
from app.models.schemas import Guidance


class HttpHunyuan3DAdapter(Hunyuan3DAdapter):
    def __init__(self, endpoint: str, timeout: float = 600.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def _load_mesh(self, payload: dict) -> trimesh.Trimesh:
        if "glb_base64" in payload:
            data = base64.b64decode(payload["glb_base64"])
            return trimesh.load(io.BytesIO(data), file_type="glb", force="mesh")
        if "mesh_url" in payload:
            resp = self._client.get(payload["mesh_url"])
            resp.raise_for_status()
            return trimesh.load(
                io.BytesIO(resp.content), file_type="glb", force="mesh"
            )
        raise ValueError("服务未返回 mesh")

    @staticmethod
    def _texture(payload: dict) -> TextureDescriptor:
        t = payload.get("texture", {})
        return TextureDescriptor(
            patterns=t.get("patterns", []),
            glossy=t.get("glossy", False),
            metalness=t.get("metalness", 0.0),
            roughness=t.get("roughness", 1.0),
        )

    def generate(
        self, images, prompt, constraints: TextConstraints, guidance: Guidance
    ) -> GenerationResult:
        resp = self._client.post(
            f"{self.endpoint}/generate",
            json={
                "images": [img.model_dump() for img in images],
                "prompt": prompt,
                "image_scale": guidance.image_scale,
                "text_scale": guidance.text_scale,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        provided = {img.view.value for img in images}
        return GenerationResult(
            mesh=self._load_mesh(data),
            texture=self._texture(data),
            source="joint",
            provided_views=provided,
            guidance=guidance,
        )

    def build_reference(self, constraints: TextConstraints) -> GenerationResult:
        resp = self._client.post(
            f"{self.endpoint}/reference", json={"prompt": constraints.raw}
        )
        resp.raise_for_status()
        data = resp.json()
        return GenerationResult(
            mesh=self._load_mesh(data), texture=self._texture(data), source="reference"
        )

    def refine_geometry(self, result, focus, reference, constraints) -> GenerationResult:
        resp = self._client.post(
            f"{self.endpoint}/refine",
            json={"focus": focus, "prompt": constraints.raw},
        )
        resp.raise_for_status()
        data = resp.json()
        return GenerationResult(
            mesh=self._load_mesh(data),
            texture=result.texture,
            source=result.source,
            provided_views=result.provided_views,
        )

    def repaint(self, result, focus, constraints) -> GenerationResult:
        resp = self._client.post(
            f"{self.endpoint}/repaint",
            json={"focus": focus, "prompt": constraints.raw},
        )
        resp.raise_for_status()
        data = resp.json()
        return GenerationResult(
            mesh=result.mesh,
            texture=self._texture(data),
            source=result.source,
            provided_views=result.provided_views,
        )
