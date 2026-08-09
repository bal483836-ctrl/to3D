"""Mock 混元 3D 适配器：无 GPU 也能产出真实网格，用于本地开发与测试。

模拟「图文联合条件生成」：图与文同时作为条件，一次生成一份网格。双模态引导
权重(guidance)决定各模态影响。当文字引导偏弱、且某信息在图像中不可得时，会
出现典型欠还原缺陷，从而让一致性自检与定向修正有真实数据可跑、可收敛：

  1. 未上传底图 且 文字引导不足 → 底部欠还原（平底）。
  2. 文字引导不足 → 花纹缺失、釉面高光被当作哑光。

自检发现后，refine/repaint 相当于提高对应维度的文字引导并局部重生成/重绘。
真实接入时用 HttpHunyuan3DAdapter 替换本类即可，接口完全一致。
"""
from __future__ import annotations

import numpy as np
import trimesh

from app.adapters.base import GenerationResult, Hunyuan3DAdapter, TextureDescriptor
from app.core.preprocess import TextConstraints
from app.models.schemas import Guidance

# 文字引导权重超过该阈值，才足以让「图像中不可得」的信息被联合生成采纳
_TEXT_APPLY_THRESHOLD = 0.6

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

    def generate(
        self,
        images: list,
        prompt: str,
        constraints: TextConstraints,
        guidance: Guidance,
    ) -> GenerationResult:
        """图文联合条件生成：图与文同时作为条件，单次生成一份网格 + 纹理。"""
        provided = {getattr(img, "view", getattr(img, "value", str(img))) for img in images}
        provided = {v.value if hasattr(v, "value") else v for v in provided}
        has_bottom = "bottom" in provided
        text_strong = guidance.text_scale >= _TEXT_APPLY_THRESHOLD

        # 体型/比例：图像可见 + 文字一致 → 联合还原良好
        # 底部：图像可见(有底图) → 采用；否则需足够文字引导才能采纳文字规格
        if has_bottom or text_strong:
            bottom = constraints.bottom_type or "flat"
        else:
            bottom = "flat"  # 图像不可得 + 文字引导不足 → 欠还原
        mesh = _build_mesh(constraints, bottom)

        # 纹理/材质：花纹与釉面高光在图像中常不可靠，需足够文字引导才被联合采纳
        metal, rough = _material_pbr(constraints)
        if text_strong:
            patterns = list(constraints.patterns)
            glossy = bool(constraints.glossy)
            rough = min(rough, 0.15) if constraints.glossy else rough
        else:
            patterns = []
            glossy = False
            rough = max(rough, 0.55)
        tex = TextureDescriptor(
            patterns=patterns, glossy=glossy, metalness=metal, roughness=rough
        )
        return GenerationResult(
            mesh=mesh, texture=tex, source="joint",
            provided_views=provided, guidance=guidance,
        )

    def build_reference(self, constraints: TextConstraints) -> GenerationResult:
        """规格重建参照（仅用于自检，不交付）。"""
        bottom = constraints.bottom_type or "flat"
        mesh = _build_mesh(constraints, bottom)
        metal, rough = _material_pbr(constraints)
        tex = TextureDescriptor(
            patterns=list(constraints.patterns),
            glossy=bool(constraints.glossy),
            metalness=metal,
            roughness=min(rough, 0.15) if constraints.glossy else rough,
        )
        return GenerationResult(mesh=mesh, texture=tex, source="reference")

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
