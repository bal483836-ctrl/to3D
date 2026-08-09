"""比对引擎的真实几何/语义度量测试。"""
from __future__ import annotations

from app.adapters.mock import MockHunyuan3DAdapter, _build_mesh
from app.core import comparison
from app.core.preprocess import TextConstraints, extract_constraints
from app.models.schemas import Dimension, FusionConfig, ImageInput, ViewName


def _imgs(views):
    return [ImageInput(view=v, url="x") for v in views]


def test_extract_constraints_chinese():
    c = extract_constraints("青花瓷花瓶，细长颈，圈足，缠枝莲纹，釉面高光")
    assert c.category == "花瓶"
    assert "ceramic" in c.materials
    assert "缠枝莲" in c.patterns
    assert c.bottom_type == "ring_foot"
    assert c.proportion_hint == "tall"
    assert c.glossy is True


def test_bottom_type_detection_from_mesh():
    c = TextConstraints()
    flat = _build_mesh(c, "flat")
    ring = _build_mesh(c, "ring_foot")
    assert comparison._bottom_type(flat) == "flat"
    assert comparison._bottom_type(ring) == "ring_foot"


def test_proportion_affects_height():
    tall = _build_mesh(TextConstraints(proportion_hint="tall"), "flat")
    wide = _build_mesh(TextConstraints(proportion_hint="wide"), "flat")
    ht, _ = comparison._extents_hw(tall)
    hw, _ = comparison._extents_hw(wide)
    assert ht > hw


def test_shape_score_identical_high():
    m = _build_mesh(TextConstraints(proportion_hint="tall"), "ring_foot")
    from app.adapters.base import GenerationResult, TextureDescriptor

    a = GenerationResult(mesh=m, texture=TextureDescriptor(), source="image")
    b = GenerationResult(mesh=m.copy(), texture=TextureDescriptor(), source="text")
    s, _ = comparison._score_shape(a, b, TextConstraints())
    assert s > 0.9


def test_compare_flags_missing_pattern_and_material():
    adapter = MockHunyuan3DAdapter()
    prompt = "青花瓷花瓶，细长颈，圈足，缠枝莲纹，釉面高光"
    c = extract_constraints(prompt)
    # 未上传底图 → 底部会被误判；花纹/材质在图通道缺失
    imgs = _imgs([ViewName.front, ViewName.left45, ViewName.right])
    a = adapter.image_to_3d(imgs, prompt, c)
    b = adapter.text_to_3d(prompt, c)
    report = comparison.compare(a, b, c, FusionConfig(consistency_threshold=0.8))

    assert not report.passed
    assert "repaint_pattern" in report.actions
    assert "adjust_material" in report.actions
    assert "refine_bottom" in report.actions  # 无底图 → 以文校正
    # 有正/侧图 → 器型以图为准，且形状相近应达标
    assert report.dimension(Dimension.shape).passed


def test_authority_arbitration_bottom_with_image():
    """上传了底图时，底部偏差应以图为准，不触发以文修正。"""
    adapter = MockHunyuan3DAdapter()
    prompt = "陶瓷花瓶，圈足"
    c = extract_constraints(prompt)
    imgs = _imgs([ViewName.front, ViewName.bottom])  # 含底图
    a = adapter.image_to_3d(imgs, prompt, c)
    b = adapter.text_to_3d(prompt, c)
    report = comparison.compare(a, b, c, FusionConfig(consistency_threshold=0.8))
    # 有底图时底部由图正确重建 → 通过；不应出现 refine_bottom 动作
    assert "refine_bottom" not in report.actions
