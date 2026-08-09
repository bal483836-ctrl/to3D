"""六维一致性自检引擎（方案 §5，产品核心）。

对**联合生成的单一产物** output 做一致性自检：度量它相对「规格参照 reference」
（由文字约束重建，仅供打分）与文字约束本身在各维度的偏离，产出可解释 DiffReport。
注意：这不是「两个模型互比」，reference 不作为交付物，只作为验证尺子。
- 几何维度(shape/height/width/bottom)：直接在网格上做真实测量。
- 语义维度(pattern/material)：基于纹理描述与文字约束评分（生产环境由 CLIP/VQA/PBR 分析得到描述）。
- 冲突仲裁(§5.4)：某维度有对应视角图 → 以图为准(接受)；无图但文字明确 → 以文校正。
"""
from __future__ import annotations

import numpy as np
import trimesh

from app.adapters.base import GenerationResult
from app.core.preprocess import TextConstraints
from app.models.schemas import (
    DIMENSION_LABELS,
    DiffReport,
    Dimension,
    DimensionResult,
    FusionConfig,
)

# Chamfer 距离 → 相似度的映射系数（对归一化点云标定：完全一致 ≈ 0.95）
_SHAPE_K = 4.0

# --- 几何工具 ----------------------------------------------------------------


def _normalize_points(mesh: trimesh.Trimesh, n: int = 2000) -> np.ndarray:
    """在网格表面采样并归一化到单位包围盒中心，用于形状比对（去除绝对尺度）。"""
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    pts = np.asarray(pts, dtype=float)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    scale = float(np.max(mesh.extents)) or 1.0
    return (pts - center) / scale


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """对称 Chamfer 距离（numpy 暴力最近邻，点数已下采样，足够快）。"""
    # a: (Na,3), b: (Nb,3)
    def _one_way(x: np.ndarray, y: np.ndarray) -> float:
        # 分块计算避免大矩阵
        d_sum = 0.0
        for chunk in np.array_split(x, max(1, len(x) // 500)):
            d = np.linalg.norm(chunk[:, None, :] - y[None, :, :], axis=2)
            d_sum += float(d.min(axis=1).sum())
        return d_sum / len(x)

    return 0.5 * (_one_way(a, b) + _one_way(b, a))


def _shape_score(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """器型相似度：Chamfer 距离映射到 0~1（越近越高）。"""
    pa, pb = _normalize_points(a), _normalize_points(b)
    cd = _chamfer(pa, pb)
    # 归一化点云的 Chamfer 距离通常在 [0, ~0.3]，用指数映射到相似度
    return float(np.exp(-cd * _SHAPE_K))


def _extents_hw(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """返回 (height, width)：height 取 Z 向，width 取 XY 最大跨度。"""
    ex, ey, ez = mesh.extents
    return float(ez), float(max(ex, ey))


def _bottom_min_radius(mesh: trimesh.Trimesh) -> float:
    """底部最小半径 / 最大半径。近 0 → 平底/尖底；显著 >0 → 圈足/空心底。"""
    v = mesh.vertices
    zmin = v[:, 2].min()
    slab = v[np.abs(v[:, 2] - zmin) < (mesh.extents[2] * 0.03 + 1e-6)]
    if len(slab) == 0:
        return 0.0
    center_xy = (mesh.bounds[0][:2] + mesh.bounds[1][:2]) / 2.0
    radii = np.linalg.norm(slab[:, :2] - center_xy, axis=1)
    rmax = float(max(mesh.extents[0], mesh.extents[1]) / 2.0) or 1.0
    return float(radii.min() / rmax)


def _bottom_type(mesh: trimesh.Trimesh) -> str:
    return "ring_foot" if _bottom_min_radius(mesh) > 0.12 else "flat"


def _symmetry_error(mesh: trimesh.Trimesh) -> float:
    """左右对称性：沿 X 镜像后与原点云的 Chamfer，越小越对称。"""
    p = _normalize_points(mesh, 1200)
    pm = p.copy()
    pm[:, 0] = -pm[:, 0]
    return _chamfer(p, pm)


# --- 逐维度评分 --------------------------------------------------------------


def _score_shape(a: GenerationResult, b: GenerationResult, c: TextConstraints):
    s = _shape_score(a.mesh, b.mesh)
    return s, f"器型 Chamfer 相似度 {s:.2f}"


def _score_height(a: GenerationResult, b: GenerationResult, c: TextConstraints):
    ha, _ = _extents_hw(a.mesh)
    hb, _ = _extents_hw(b.mesh)
    # 以文通道高度为参照的相对偏差
    rel = abs(ha - hb) / (max(ha, hb) or 1.0)
    s = float(max(0.0, 1.0 - rel))
    return s, f"高度 A={ha:.2f} B={hb:.2f} 相对偏差 {rel:.2f}"


def _score_width(a: GenerationResult, b: GenerationResult, c: TextConstraints):
    _, wa = _extents_hw(a.mesh)
    _, wb = _extents_hw(b.mesh)
    rel = abs(wa - wb) / (max(wa, wb) or 1.0)
    s_prop = max(0.0, 1.0 - rel)
    detail = f"宽度 A={wa:.2f} B={wb:.2f} 相对偏差 {rel:.2f}"
    if c.symmetry:
        sym_err = _symmetry_error(a.mesh)
        s_sym = float(np.exp(-sym_err * _SHAPE_K))
        detail += f"；对称性 {s_sym:.2f}"
        return float(min(s_prop, s_sym)), detail
    return float(s_prop), detail


def _score_bottom(a: GenerationResult, b: GenerationResult, c: TextConstraints):
    ta, tb = _bottom_type(a.mesh), _bottom_type(b.mesh)
    target = c.bottom_type or tb  # 文字明确则以文字为目标，否则以文通道为目标
    match = (ta == target)
    s = 1.0 if match else 0.35
    return s, f"底部 A={ta} 目标={target}"


def _score_pattern(a: GenerationResult, b: GenerationResult, c: TextConstraints):
    target = set(c.patterns) if c.patterns else set(b.texture.patterns)
    got = set(a.texture.patterns)
    if not target:
        return 1.0, "文字/参照均未要求特定花纹"
    inter = target & got
    s = len(inter) / len(target)
    missing = target - got
    detail = f"花纹匹配 {len(inter)}/{len(target)}"
    if missing:
        detail += f"，缺失 {sorted(missing)}"
    return float(s), detail


def _score_material(a: GenerationResult, b: GenerationResult, c: TextConstraints):
    # 金属度/粗糙度接近度
    dm = abs(a.texture.metalness - b.texture.metalness)
    dr = abs(a.texture.roughness - b.texture.roughness)
    s_pbr = max(0.0, 1.0 - (dm + dr) / 2.0)
    # 光泽意图一致性（文字明确时优先）
    target_glossy = c.glossy if c.glossy is not None else b.texture.glossy
    gloss_ok = (a.texture.glossy == target_glossy)
    s = s_pbr * (1.0 if gloss_ok else 0.5)
    detail = f"PBR 接近度 {s_pbr:.2f}；光泽 {'一致' if gloss_ok else '不一致'}"
    return float(s), detail


_SCORERS = {
    Dimension.shape: _score_shape,
    Dimension.height: _score_height,
    Dimension.width: _score_width,
    Dimension.bottom: _score_bottom,
    Dimension.pattern: _score_pattern,
    Dimension.material: _score_material,
}

# 几何维度：图片可直接验证，有对应视角图时以图为准（保护图验证过的几何）。
_GEOMETRY_DIMS = {Dimension.shape, Dimension.height, Dimension.width, Dimension.bottom}
# 纹理/材质维度：属于表面细节，图生3D 常欠还原，偏差一律按约束重绘。
_TEXTURE_DIMS = {Dimension.pattern, Dimension.material}

# 几何维度对应的视角图（用于冲突仲裁：有图→以图为准，不被文字带偏）
_DIM_VIEWS: dict[Dimension, set[str]] = {
    Dimension.bottom: {"bottom"},
    Dimension.shape: {"front", "left", "right", "back"},
    Dimension.height: {"front", "left", "right", "back"},
    Dimension.width: {"front", "left", "right", "back"},
}


def _arbitrate(
    dim: Dimension, provided_views: set[str], c: TextConstraints, passed: bool
) -> tuple[str, bool]:
    """冲突仲裁（方案 §5.4）。返回 (authority, need_correction)。

    - 达标：以图为准，无需修正。
    - 纹理/材质偏差：一律修正（目标来自文字，否则来自文通道参照）。
    - 几何偏差：有对应视角图 → 以图为准且不修正（接受图验证过的差异）；
      无图但文字明确 → 以文校正；两者皆弱 → 低置信，不自动修正。
    """
    if passed:
        return "image", False
    if dim in _TEXTURE_DIMS:
        return ("text" if c.mentions(dim.value) else "image"), True
    # 几何维度
    has_view = bool(_DIM_VIEWS.get(dim, set()) & provided_views)
    if has_view:
        return "image", False  # 图验证过的几何，接受与文字的差异
    if c.mentions(dim.value):
        return "text", True
    return "low_confidence", False


_ACTION_BY_DIM = {
    Dimension.pattern: "repaint_pattern",
    Dimension.material: "adjust_material",
    Dimension.bottom: "refine_bottom",
    Dimension.shape: "refine_shape",
    Dimension.height: "refine_proportion",
    Dimension.width: "refine_proportion",
}


def compare(
    output: GenerationResult,
    reference: GenerationResult,
    constraints: TextConstraints,
    fusion: FusionConfig,
) -> DiffReport:
    """对联合产物 output 执行六维一致性自检，产出 DiffReport。

    reference 为规格重建参照，仅用于几何/材质的偏离度量，不作为交付物。
    """
    weights = fusion.weights()
    threshold = fusion.consistency_threshold
    provided = output.provided_views

    results: list[DimensionResult] = []
    actions: list[str] = []
    overall = 0.0

    for dim, scorer in _SCORERS.items():
        score, detail = scorer(output, reference, constraints)
        passed = score >= threshold
        authority, need_fix = _arbitrate(dim, provided, constraints, passed)
        results.append(
            DimensionResult(
                dimension=dim,
                label=DIMENSION_LABELS[dim],
                score=round(score, 4),
                passed=passed,
                detail=detail,
                authority=authority,
            )
        )
        overall += weights[dim] * score
        if need_fix:
            actions.append(_ACTION_BY_DIM[dim])

    overall = round(overall, 4)
    # 报告整体通过 = 所有「需以文校正」的偏差都不存在（即无待修正动作）且总分达标
    report_passed = (len(actions) == 0) and (overall >= threshold)
    return DiffReport(
        overall=overall,
        threshold=threshold,
        passed=report_passed,
        dimensions=results,
        actions=sorted(set(actions)),
    )
