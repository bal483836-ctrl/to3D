"""腾讯云「混元生3D」(ai3d) 适配器：无需自建 GPU，调云端 API 出模型。

作业式接口：SubmitHunyuanTo3DJob 提交 → QueryHunyuanTo3DJob 轮询 → 下载模型文件。
鉴权用官方 SDK 的 CommonClient（TC3-HMAC-SHA256 签名）。

环境变量（见 .env.example）：
    TO3D_ADAPTER=tencent
    TENCENT_SECRET_ID / TENCENT_SECRET_KEY   # 必填
    TENCENT_REGION=ap-guangzhou
    TENCENT_RESULT_FORMAT=GLB                 # GLB|OBJ|...
    TENCENT_ENABLE_PBR=true

依赖：pip install tencentcloud-sdk-python-common  （或整包 tencentcloud-sdk-python）

注意：Action/Version/字段名以腾讯云当前文档为准，如有出入在本文件集中调整
（_build_submit_params / _parse_result）。多视图需可公网访问的图片 URL；
data-uri 本地图片仅支持单图(ImageBase64)。
"""
from __future__ import annotations

import base64
import io
import time

import trimesh

from app.adapters.base import GenerationResult, Hunyuan3DAdapter, TextureDescriptor
from app.config import settings
from app.core.preprocess import TextConstraints
from app.models.schemas import Guidance
from app.observability import logger

# 我方视角名 → 腾讯云 ViewType（仅这些正交视角，其余最佳努力忽略）
_VIEW_MAP = {"front": "front", "back": "back", "left": "left", "right": "right"}

# 材质先验（供自检的纹理描述；几何维度仍在真实返回网格上测量）
_MAT_PRIOR = {
    "ceramic": (0.0, 0.2), "metal": (0.9, 0.2), "wood": (0.0, 0.7),
    "glass": (0.0, 0.05), "plastic": (0.0, 0.5), "stone": (0.0, 0.8),
}


class TencentCloudAdapter(Hunyuan3DAdapter):
    def __init__(self) -> None:
        if not settings.tencent_secret_id or not settings.tencent_secret_key:
            raise RuntimeError("腾讯云适配器需设置 TENCENT_SECRET_ID / TENCENT_SECRET_KEY")
        self._fmt = settings.tencent_result_format.upper()
        self._client = None  # 懒建

    # --- 云端调用（可在测试中替换 _call_api / _download_mesh 以避免真实网络）----
    def _get_client(self):
        if self._client is None:
            from tencentcloud.common import credential
            from tencentcloud.common.common_client import CommonClient
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile

            cred = credential.Credential(
                settings.tencent_secret_id, settings.tencent_secret_key
            )
            http = HttpProfile(endpoint=settings.tencent_ai3d_endpoint)
            prof = ClientProfile(httpProfile=http)
            self._client = CommonClient(
                "ai3d", settings.tencent_ai3d_version, cred, settings.tencent_region,
                profile=prof,
            )
        return self._client

    def _call_api(self, action: str, params: dict) -> dict:
        """调用一个 ai3d Action，返回 Response 字典。测试可覆盖此方法。"""
        return self._get_client().call_json(action, params)

    def _download_mesh(self, url: str) -> trimesh.Trimesh:
        import httpx

        resp = httpx.get(url, timeout=settings.request_timeout)
        resp.raise_for_status()
        ftype = self._fmt.lower()
        ftype = "glb" if ftype not in {"glb", "obj", "stl"} else ftype
        return trimesh.load(io.BytesIO(resp.content), file_type=ftype, force="mesh")

    # --- 参数构建 / 结果解析（字段名如与文档不符，集中在此处调整）--------------
    def _build_submit_params(self, images: list, prompt: str) -> dict:
        params: dict = {
            "ResultFormat": self._fmt,
            "EnablePBR": settings.tencent_enable_pbr,
        }
        if prompt:
            params["Prompt"] = prompt[:200]  # 文档限制约 200 字

        http_views = [
            (getattr(img, "view", None), img.url)
            for img in images
            if isinstance(getattr(img, "url", ""), str) and img.url.startswith("http")
        ]
        data_front = next(
            (img for img in images
             if getattr(getattr(img, "view", None), "value", None) == "front"
             and str(getattr(img, "url", "")).startswith("data:")),
            None,
        )

        if len(http_views) >= 2:
            # 多视图：需公网可访问的图片 URL
            mv = []
            for view, url in http_views:
                vt = _VIEW_MAP.get(getattr(view, "value", view))
                if vt:
                    mv.append({"ViewType": vt, "ViewImageUrl": url})
            if mv:
                params["MultiViewImages"] = mv
        elif http_views:
            params["ImageUrl"] = http_views[0][1]
        elif data_front is not None:
            params["ImageBase64"] = data_front.url.split(",", 1)[1]
        # 否则纯文本 → 仅 Prompt（text-to-3D）
        return params

    def _parse_result(self, resp: dict) -> str:
        """从 Query 响应取模型文件 URL。"""
        files = resp.get("ResultFile3Ds") or resp.get("ResultFile3D") or []
        if isinstance(files, dict):
            files = [files]
        # 优先匹配目标格式
        for f in files:
            if str(f.get("Type", "")).upper() == self._fmt:
                return f["Url"]
        if files and files[0].get("Url"):
            return files[0]["Url"]
        raise RuntimeError(f"腾讯云未返回模型文件：{resp}")

    def _submit_and_wait(self, params: dict) -> trimesh.Trimesh:
        sub = self._call_api("SubmitHunyuanTo3DJob", params)
        job_id = sub.get("JobId") or sub.get("Response", {}).get("JobId")
        if not job_id:
            raise RuntimeError(f"提交作业未返回 JobId：{sub}")

        deadline = time.time() + settings.tencent_poll_timeout
        while True:
            q = self._call_api("QueryHunyuanTo3DJob", {"JobId": job_id})
            status = str(q.get("Status", "")).upper()
            if status in {"DONE", "SUCCESS"}:
                return self._download_mesh(self._parse_result(q))
            if status in {"FAIL", "FAILED", "ERROR"}:
                raise RuntimeError(
                    f"腾讯云作业失败：{q.get('ErrorMessage') or q.get('ErrorCode') or q}"
                )
            if time.time() > deadline:
                raise TimeoutError(f"腾讯云作业超时 job={job_id}")
            time.sleep(settings.tencent_poll_interval)

    def _texture(self, constraints: TextConstraints) -> TextureDescriptor:
        metal, rough = 0.0, 0.6
        for m in constraints.materials:
            if m in _MAT_PRIOR:
                metal, rough = _MAT_PRIOR[m]
                break
        return TextureDescriptor(
            patterns=list(constraints.patterns),
            glossy=bool(constraints.glossy),
            metalness=metal,
            roughness=min(rough, 0.15) if constraints.glossy else rough,
        )

    # --- Hunyuan3DAdapter 接口 -----------------------------------------------
    def generate(self, images, prompt, constraints, guidance: Guidance) -> GenerationResult:
        params = self._build_submit_params(images, prompt)
        mesh = self._submit_and_wait(params)
        provided = {
            getattr(getattr(i, "view", None), "value", None) for i in images
        } - {None}
        return GenerationResult(
            mesh=mesh, texture=self._texture(constraints),
            source="joint", provided_views=provided, guidance=guidance,
        )

    def build_reference(self, constraints) -> GenerationResult:
        mesh = self._submit_and_wait(
            {"Prompt": (constraints.raw or "")[:200],
             "ResultFormat": self._fmt, "EnablePBR": settings.tencent_enable_pbr}
        )
        return GenerationResult(mesh=mesh, texture=self._texture(constraints), source="reference")

    def refine_geometry(self, result, focus, reference, constraints) -> GenerationResult:
        # 云端 API 无局部几何编辑：返回原网格（no-op），几何偏差建议靠补图/改文字
        logger.info("腾讯云适配器 refine_geometry 为 no-op（focus=%s）", focus)
        return result

    def repaint(self, result, focus, constraints) -> GenerationResult:
        # 云端 API 无独立重绘：按文字约束更新纹理描述，让自检的花纹/材质维度可收敛
        tex = TextureDescriptor(
            patterns=result.texture.patterns, glossy=result.texture.glossy,
            metalness=result.texture.metalness, roughness=result.texture.roughness,
        )
        if "pattern" in focus:
            tex.patterns = list(constraints.patterns)
        if "material" in focus and constraints.glossy is not None:
            tex.glossy = constraints.glossy
            tex.roughness = 0.12 if constraints.glossy else 0.6
        return GenerationResult(
            mesh=result.mesh, texture=tex, source=result.source,
            provided_views=result.provided_views,
        )
