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


# 文字数值尺寸换算出的高宽比允许区间（挡住抽取异常导致的畸形网格）
_ASPECT_MIN, _ASPECT_MAX = 0.2, 6.0

# 三足规格：足数、足半径、足所在圆周半径、足高占总高比例
_TRIPOD_LEGS = 3
_LEG_RADIUS, _LEG_CIRCLE, _LEG_H_FRAC = 0.09, 0.28, 0.22


def _target_height(constraints: TextConstraints) -> float:
    """文字规格给出的归一化高度（最大直径归一为 1.0）。

    文字给了「高 / 口径」两个数值时按真实高宽比建模；否则回落到比例意图词。
    """
    aspect = constraints.aspect()
    if aspect is not None:
        return float(min(max(aspect, _ASPECT_MIN), _ASPECT_MAX))
    return _proportion_height(constraints.proportion_hint)


def _body_profile(H: float, z0: float, rmax: float = 0.5) -> list[tuple[float, float]]:
    """器身母线：腹径 → 收腰 → 细颈 → 口沿 → 顶盖回到轴心。

    各拐点按总高的固定比例排布（与底部形态无关），腹部略高于底部段末端 z0，
    保证母线的 z 单调递增。
    """
    neck, rim = 0.18, 0.22
    z_belly = max(0.35 * H, z0 + 0.02 * H)
    return [
        (rmax, z_belly), (rmax * 0.62, 0.60 * H), (neck, 0.82 * H),
        (rim, 0.97 * H), (rim * 0.9, H), (0.0, H),
    ]


def _vase_profile(height: float, bottom_style: str) -> np.ndarray:
    """生成回转母线 (radius, z)，绕 Z 轴回转成实体。底部形态决定起始段。"""
    H = height
    rmax = 0.5
    pts: list[tuple[float, float]] = []
    if bottom_style == "ring_foot":
        # 圈足：底部中心内凹，靠外圈环着地
        r_outer, r_inner, h_recess = 0.30, 0.17, 0.06 * H
        pts += [
            (0.0, h_recess), (r_inner, h_recess), (r_inner, 0.0),
            (r_outer, 0.0), (r_outer, 0.05 * H), (0.36, 0.11 * H),
        ]
        z0 = 0.11 * H
    elif bottom_style == "pointed":
        # 尖底：自轴心一点起锥形张开，着地面积近乎为零
        pts += [(0.0, 0.0), (0.18, 0.10 * H), (0.34, 0.20 * H)]
        z0 = 0.20 * H
    elif bottom_style == "round":
        # 圆底：四分之一椭圆弧，着地为一小片球冠，半径随高度快速增大
        arc_h = 0.30 * H
        for t in np.linspace(0.0, np.pi / 2.0, 10)[:-1]:
            pts.append((float(rmax * np.sin(t)), float(arc_h * (1.0 - np.cos(t)))))
        z0 = arc_h
    elif bottom_style == "base":
        # 底座：外扩的台座，向上收一级台阶后才接器身
        pts += [
            (0.0, 0.0), (0.62, 0.0), (0.62, 0.045 * H),
            (0.40, 0.075 * H), (0.36, 0.13 * H),
        ]
        z0 = 0.13 * H
    else:  # flat / 其它 → 平底
        pts += [(0.0, 0.0), (rmax * 0.72, 0.0)]
        z0 = 0.0
    pts += _body_profile(H, z0, rmax)
    return np.array(pts, dtype=float)


def _build_tripod(height: float) -> trimesh.Trimesh:
    """三足器（鬲/鼎形）：圆底器身 + 三条等分立足，足底着地。"""
    H = height
    leg_h = _LEG_H_FRAC * H
    body_h = H - leg_h
    # 器身：圆底钵体，整体抬到足高之上
    profile = _vase_profile(body_h, "round")
    body = trimesh.creation.revolve(profile, sections=64)
    body.apply_translation([0.0, 0.0, leg_h - body.bounds[0][2]])

    parts = [body]
    for i in range(_TRIPOD_LEGS):
        ang = 2.0 * np.pi * i / _TRIPOD_LEGS + np.pi / 2.0
        # 足略微上探进器身，保证外观相接
        leg = trimesh.creation.cylinder(
            radius=_LEG_RADIUS, height=leg_h * 1.25, sections=24
        )
        leg.apply_translation([
            _LEG_CIRCLE * np.cos(ang), _LEG_CIRCLE * np.sin(ang), leg_h * 1.25 / 2.0,
        ])
        parts.append(leg)
    return trimesh.util.concatenate(parts)


def _build_mesh(
    constraints: TextConstraints,
    bottom_style: str,
    height: float | None = None,
) -> trimesh.Trimesh:
    """按底部形态与目标高度构网格。height 为 None 时用比例意图词推导。"""
    if height is None:
        height = _proportion_height(constraints.proportion_hint)
    if bottom_style == "tripod":
        mesh = _build_tripod(height)
    else:
        mesh = trimesh.creation.revolve(_vase_profile(height, bottom_style), sections=64)
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
        # 精确数值尺寸(高/口径)是文字独有信息——图像只给得出比例、给不出绝对尺寸，
        # 因此需足够的文字引导才会被联合生成采纳，否则沿用图像推得的默认比例。
        height = (
            _target_height(constraints)
            if text_strong
            else _proportion_height(constraints.proportion_hint)
        )
        mesh = _build_mesh(constraints, bottom, height)

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
        mesh = _build_mesh(constraints, bottom, _target_height(constraints))
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
        # 依据文字约束重建被误判的几何区域（底部形态 / 比例尺寸）
        bottom = constraints.bottom_type or "flat"
        if {"height", "width", "shape"} & set(focus):
            height = _target_height(constraints)  # 按文字规格重建比例
        else:
            # 非比例维度的修正不应改动已有比例，沿用当前产物的归一化高宽比
            ex = result.mesh.extents
            height = float(ex[2] / (max(ex[0], ex[1]) or 1.0))
        mesh = _build_mesh(constraints, bottom, height)
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
