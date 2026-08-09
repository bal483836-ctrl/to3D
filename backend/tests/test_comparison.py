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


def test_joint_generation_self_check_flags_undersatisfied():
    """图文联合生成(图像引导为主)时，图像不可得且文字引导偏弱的信息欠还原，
    自检应逐项标出并给出定向修正动作。"""
    adapter = MockHunyuan3DAdapter()
    prompt = "青花瓷花瓶，细长颈，圈足，缠枝莲纹，釉面高光"
    c = extract_constraints(prompt)
    fc = FusionConfig(strategy="image_primary", consistency_threshold=0.8)
    imgs = _imgs([ViewName.front, ViewName.left45, ViewName.right])  # 无底图
    out = adapter.generate(imgs, prompt, c, fc.guidance())
    ref = adapter.build_reference(c)
    report = comparison.compare(out, ref, c, fc)

    assert not report.passed
    assert "repaint_pattern" in report.actions
    assert "adjust_material" in report.actions
    assert "refine_bottom" in report.actions  # 无底图 + 文字引导弱 → 以文校正
    # 有正/侧图 → 器型以图为准，形状应达标
    assert report.dimension(Dimension.shape).passed


def test_text_strong_guidance_satisfies_in_one_pass():
    """文字引导更强时，联合生成一次即应满足文字约束（无待修正动作）。"""
    adapter = MockHunyuan3DAdapter()
    prompt = "青花瓷花瓶，细长颈，圈足，缠枝莲纹，釉面高光"
    c = extract_constraints(prompt)
    fc = FusionConfig(strategy="text_correct", consistency_threshold=0.8)
    imgs = _imgs([ViewName.front, ViewName.left45, ViewName.right])
    out = adapter.generate(imgs, prompt, c, fc.guidance())
    ref = adapter.build_reference(c)
    report = comparison.compare(out, ref, c, fc)
    assert report.actions == []
    assert report.passed


def test_authority_arbitration_bottom_with_image():
    """上传了底图时，底部由图像直接还原，不触发以文修正。"""
    adapter = MockHunyuan3DAdapter()
    prompt = "陶瓷花瓶，圈足"
    c = extract_constraints(prompt)
    fc = FusionConfig(strategy="image_primary", consistency_threshold=0.8)
    imgs = _imgs([ViewName.front, ViewName.bottom])  # 含底图
    out = adapter.generate(imgs, prompt, c, fc.guidance())
    ref = adapter.build_reference(c)
    report = comparison.compare(out, ref, c, fc)
    assert "refine_bottom" not in report.actions
