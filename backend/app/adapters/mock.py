"""Mock 混元 3D 适配器：无 GPU 也能产出真实网格，用于本地开发与测试。

它以「文字约束」为标准答案构造一个理想网格(文通道)，并模拟图通道的两类
典型缺陷，从而让比对引擎与修正闭环有真实数据可跑、可收敛：

  1. 缺失视角 → 底部被误判为平底（若未上传底图）。
  2. 纹理/材质在图片中不可靠 → 花纹缺失、釉面高光被当作哑光。

修正阶段(refine_geometry / repaint)按文字约束把这些缺陷拉回一致。
真实接入时用 HttpHunyuan3DAdapter 替换本类即可，接口完全一致。
"""
from __future__ import annotations

import numpy as np
import trimesh

from app.adapters.base import GenerationResult, Hunyuan3DAdapter, TextureDescriptor
from app.core.preprocess import TextConstraints

# 材质 → (metalness, roughness) 先验
_MATERIAL_PBR = {
    "ceramic": (0.0, 0.25),
    "metal": (0.9, 0.2),
    "wood": (0.0, 0.7),
    "glass": (0.0, 0.05),
    "plastic": (0.0, 0.5),
    "stone": (0.0, 0.8),
}


def _proportion_height(hint: str | None) -> float:
    """比例意图 → 归一化高度（最大直径固定约 1.0）。"""
    return {"tall": 1.6, "wide": 0.85}.get(hint or "", 1.15)


def _vase_profile(height: float, bottom_style: str) -> np.ndarray:
    """生成花瓶回转母线 (radius, z)，绕 Z 轴回转成实体。"""
    H = height
    rmax, neck, rim = 0.5, 0.18, 0.22
    pts: list[tuple[float, float]] = []
    if bottom_style == "ring_foot":
        # 圈足：底部中心内凹，靠外圈环着地
        r_outer, r_inner, h_recess = 0.30, 0.17, 0.06 * H
        pts += [
            (0.0, h_recess), (r_inner, h_recess), (r_inner, 0.0),
            (r_outer, 0.0), (r_outer, 0.05 * H), (0.36, 0.11 * H),
        ]
    else:  # flat / 其它 → 平底
        pts += [(0.0, 0.0), (rmax * 0.72, 0.0)]
    # 瓶身 → 收腰 → 细颈 → 口沿 → 顶盖回到轴
    pts += [
        (rmax, 0.35 * H), (rmax * 0.62, 0.60 * H), (neck, 0.82 * H),
        (rim, 0.97 * H), (rim * 0.9, H), (0.0, H),
    ]
    return np.array(pts, dtype=float)


def _build_mesh(constraints: TextConstraints, bottom_style: str) -> trimesh.Trimesh:
    height = _proportion_height(constraints.proportion_hint)
    profile = _vase_profile(height, bottom_style)
    mesh = trimesh.creation.revolve(profile, sections=64)
    # 移到原点附近，底面贴 z=0
    mesh.apply_translation(-mesh.bounds[0] * np.array([0, 0, 1.0]))
    return mesh


def _material_pbr(constraints: TextConstraints) -> tuple[float, float]:
    for m in constraints.materials:
        if m in _MATERIAL_PBR:
            return _MATERIAL_PBR[m]
    return (0.0, 0.6)


class MockHunyuan3DAdapter(Hunyuan3DAdapter):
    """确定性 Mock，便于稳定测试。"""

    def text_to_3d(self, prompt: str, constraints: TextConstraints) -> GenerationResult:
        # 文通道：以文字约束为标准答案构造理想体
        bottom = constraints.bottom_type or "flat"
        mesh = _build_mesh(constraints, bottom)
        metal, rough = _material_pbr(constraints)
        tex = TextureDescriptor(
            patterns=list(constraints.patterns),
            glossy=bool(constraints.glossy),
            metalness=metal,
            roughness=rough if not constraints.glossy else min(rough, 0.15),
        )
        return GenerationResult(mesh=mesh, texture=tex, source="text")

    def image_to_3d(
        self, images: list, prompt: str, constraints: TextConstraints
    ) -> GenerationResult:
        provided = {getattr(img, "view", getattr(img, "value", str(img))) for img in images}
        provided = {v.value if hasattr(v, "value") else v for v in provided}
        has_bottom = "bottom" in provided

        # 几何：图片可见的体型/比例还原良好；底部若无底图则误判为平底
        bottom = (constraints.bottom_type or "flat") if has_bottom else "flat"
        mesh = _build_mesh(constraints, bottom)

        # 纹理：图片中花纹不可靠 → 缺失；釉面高光常因光照被当作哑光
        metal, rough = _material_pbr(constraints)
        tex = TextureDescriptor(
            patterns=[],          # 未捕获花纹
            glossy=False,         # 误判为哑光
            metalness=metal,
            roughness=max(rough, 0.55),
        )
        return GenerationResult(
            mesh=mesh, texture=tex, source="image", provided_views=provided
        )

    def refine_geometry(
        self,
        result: GenerationResult,
        focus: list[str],
        reference: GenerationResult,
        constraints: TextConstraints,
    ) -> GenerationResult:
        # 依据文字约束重建被误判的几何区域（如底部）
        bottom = constraints.bottom_type or "flat"
        mesh = _build_mesh(constraints, bottom)
        return GenerationResult(
            mesh=mesh,
            texture=result.texture,
            source=result.source,
            provided_views=result.provided_views,
        )

    def repaint(
        self,
        result: GenerationResult,
        focus: list[str],
        constraints: TextConstraints,
    ) -> GenerationResult:
        # 按文字约束重绘纹理/材质
        tex = TextureDescriptor(
            patterns=result.texture.patterns,
            glossy=result.texture.glossy,
            metalness=result.texture.metalness,
            roughness=result.texture.roughness,
        )
        if "pattern" in focus:
            tex.patterns = list(constraints.patterns)
        if "material" in focus:
            if constraints.glossy is not None:
                tex.glossy = constraints.glossy
                tex.roughness = 0.12 if constraints.glossy else 0.6
            metal, rough = _material_pbr(constraints)
            tex.metalness = metal
        return GenerationResult(
            mesh=result.mesh,
            texture=tex,
            source=result.source,
            provided_views=result.provided_views,
        )
