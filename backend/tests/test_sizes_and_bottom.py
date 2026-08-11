"""数值尺寸约束与底部形态的真实几何测量。

覆盖三件事：
1. 从中文描述里抽取「高 / 口径」并折算为厘米（且不把年份当尺寸）。
2. 六种底部形态都能被生成、并能在网格上被真实测量识别（生成→测量往返）。
3. 数值尺寸参与自检：偏差以厘米表述，且能触发定向修正并收敛。
"""
from __future__ import annotations

import pytest

from app.adapters.mock import MockHunyuan3DAdapter, _build_mesh, _target_height
from app.core import comparison, orchestrator
from app.core.preprocess import TextConstraints, extract_constraints, extract_sizes
from app.models.schemas import (
    Dimension,
    FusionConfig,
    GenerationRequest,
    ImageInput,
    TaskState,
    TaskStatus,
    ViewName,
)
from app.store import store

# 截图里的真实案例：西周三足陶器
POTTERY = "陶器，三足，下方由历史沉淀发黑，西周、公元前11世纪中叶—前711年、高17.5、口径16.5公分"

ALL_BOTTOMS = ["flat", "ring_foot", "tripod", "pointed", "round", "base"]


def _imgs(views):
    return [ImageInput(view=v, url="x") for v in views]


# --- 1. 数值尺寸抽取 ---------------------------------------------------------


def test_extract_sizes_units():
    assert extract_sizes("高17.5、口径16.5公分") == (17.5, 16.5)
    assert extract_sizes("通高 40 厘米，最大直径 10cm") == (40.0, 10.0)
    assert extract_sizes("高 35mm，口径2.4厘米") == (3.5, 2.4)
    assert extract_sizes("高0.4米，宽 0.2 米") == (40.0, 20.0)


def test_extract_sizes_ignores_years_and_absent():
    """「公元前711年」这类数字没有尺寸前缀，不能被当成尺寸。"""
    assert extract_sizes("西周，公元前11世纪中叶—前711年") == (None, None)
    assert extract_sizes("青花瓷花瓶，细长颈，圈足") == (None, None)


def test_pottery_prompt_full_extraction():
    c = extract_constraints(POTTERY)
    assert c.bottom_type == "tripod"
    assert (c.height_cm, c.width_cm) == (17.5, 16.5)
    assert c.has_numeric_size()
    assert c.aspect() == pytest.approx(17.5 / 16.5, rel=1e-6)
    assert "ceramic" in c.materials
    # 数值尺寸使高度/宽度维度成为「文字明确」的维度
    assert c.mentions("height") and c.mentions("width")


def test_aspect_drives_mesh_proportion():
    """文字给出高/口径时，网格按真实高宽比构建。"""
    c = extract_constraints("陶罐，通高 40 厘米，口径 10 厘米")
    assert _target_height(c) == pytest.approx(4.0)
    mesh = _build_mesh(c, "flat", _target_height(c))
    h, w = comparison._extents_hw(mesh)
    assert h / w == pytest.approx(4.0, rel=0.02)


# --- 2. 底部形态：生成 → 真实测量往返 ----------------------------------------


@pytest.mark.parametrize("style", ALL_BOTTOMS)
@pytest.mark.parametrize("height", [0.85, 1.15, 2.5])
def test_bottom_shape_roundtrip(style, height):
    """六种底部形态都能造出来，并能从网格上被测量识别（与比例无关）。"""
    mesh = _build_mesh(TextConstraints(), style, height)
    assert comparison._bottom_shape(mesh)[0] == style


def test_tripod_has_three_feet():
    mesh = _build_mesh(TextConstraints(), "tripod")
    shape, feet = comparison._bottom_shape(mesh)
    assert (shape, feet) == ("tripod", 3)


def test_bottom_score_grades_near_misses():
    """形态不匹配时按物理接近度给分，而非一律最低分。"""
    c = TextConstraints()
    from app.adapters.base import GenerationResult, TextureDescriptor

    def res(style):
        return GenerationResult(mesh=_build_mesh(c, style), texture=TextureDescriptor())

    ref = res("flat")
    # 平底 vs 底座：相近
    s_near, _ = comparison._score_bottom(res("base"), ref, TextConstraints(bottom_type="flat"))
    # 三足 vs 平底：完全不同
    s_far, _ = comparison._score_bottom(res("tripod"), ref, TextConstraints(bottom_type="flat"))
    s_hit, detail = comparison._score_bottom(res("flat"), ref, TextConstraints(bottom_type="flat"))
    assert s_hit == 1.0 and "平底" in detail
    assert s_far < s_near < s_hit


# --- 3. 尺寸参与自检与修正闭环 ----------------------------------------------


def test_tripod_pottery_bottom_now_matches():
    """截图里的回归：三足被正确生成并识别，底部维度不再锁死在 0.35。"""
    adapter = MockHunyuan3DAdapter()
    c = extract_constraints(POTTERY)
    fc = FusionConfig(strategy="image_primary")
    out = adapter.generate(_imgs([ViewName.front, ViewName.bottom]), POTTERY, c, fc.guidance())
    report = comparison.compare(out, adapter.build_reference(c), c, fc)

    assert comparison._bottom_shape(out.mesh) == ("tripod", 3)
    bottom = report.dimension(Dimension.bottom)
    assert bottom.score == 1.0 and bottom.passed


def test_size_deviation_reported_in_centimeters():
    """高度/宽度不再是无信息的常数自比，而是标定到厘米的真实偏差。"""
    adapter = MockHunyuan3DAdapter()
    c = extract_constraints(POTTERY)
    fc = FusionConfig(strategy="image_primary")
    out = adapter.generate(_imgs([ViewName.front]), POTTERY, c, fc.guidance())
    report = comparison.compare(out, adapter.build_reference(c), c, fc)

    detail = report.dimension(Dimension.height).detail
    assert "17.5cm" in detail and "cm" in detail
    assert report.dimension(Dimension.width).detail.count("cm") >= 2


def test_numeric_size_is_text_authority_even_with_front_view():
    """正图是必填项，若尺寸也判「以图为准」，数值规格将永远无法修正。
    照片没有比例尺 → 文字给了数值就以文校正。"""
    adapter = MockHunyuan3DAdapter()
    prompt = "陶罐，圈足，通高 40 厘米，口径 10 厘米"
    c = extract_constraints(prompt)
    fc = FusionConfig(strategy="image_primary")
    # 提供全部侧视图，仍应以文校正比例
    out = adapter.generate(
        _imgs([ViewName.front, ViewName.left, ViewName.right, ViewName.back]),
        prompt, c, fc.guidance(),
    )
    report = comparison.compare(out, adapter.build_reference(c), c, fc)

    height = report.dimension(Dimension.height)
    assert not height.passed
    assert height.authority == "text"
    assert "refine_proportion" in report.actions


@pytest.mark.asyncio
async def test_size_mismatch_converges_end_to_end():
    """细高器物：初轮按图像默认比例欠还原，一轮定向修正后收敛到文字规格。"""
    req = GenerationRequest(
        images=[
            ImageInput(view=ViewName.front, url="x", required=True),
            ImageInput(view=ViewName.bottom, url="x"),
        ],
        prompt="陶罐，圈足，通高 40 厘米，口径 10 厘米",
    )
    task_id = orchestrator.new_task_id()
    store.create(TaskState(task_id=task_id, status=TaskStatus.queued))
    await orchestrator.run_generation(task_id, req)

    state = store.get(task_id)
    assert state.status == TaskStatus.done
    report = state.outputs.final_diff_report
    assert report.actions == [] and report.passed
    assert state.round >= 1
    # 最终产物确实按 40:10 的比例交付
    assert "40cm" in report.dimension(Dimension.height).detail


@pytest.mark.asyncio
async def test_export_is_y_up_for_gltf():
    """导出必须转成 glTF 约定的 +Y 朝上，否则预览器与 Blender 里模型全部侧躺。

    内部度量用 Z-up，交付文件用 Y-up：细高器物导出后最长轴应落在 Y 上。
    """
    import trimesh

    req = GenerationRequest(
        images=[
            ImageInput(view=ViewName.front, url="x", required=True),
            ImageInput(view=ViewName.bottom, url="x"),
        ],
        prompt="陶罐，圈足，通高 40 厘米，口径 10 厘米",
    )
    task_id = orchestrator.new_task_id()
    store.create(TaskState(task_id=task_id, status=TaskStatus.queued))
    await orchestrator.run_generation(task_id, req)

    exported = trimesh.load(store.get(task_id).outputs.glb, force="mesh")
    ex, ey, ez = exported.extents
    assert ey > ex and ey > ez, f"导出的模型不是 Y-up：extents={exported.extents}"
    assert ey / max(ex, ez) == pytest.approx(4.0, rel=0.05)


@pytest.mark.asyncio
async def test_tripod_pottery_converges_without_bottom_view():
    """无底图 + 图像引导为主：三足欠还原 → 以文校正 → 收敛为真三足。"""
    req = GenerationRequest(
        images=[
            ImageInput(view=ViewName.front, url="x", required=True),
            ImageInput(view=ViewName.left45, url="x"),
        ],
        prompt=POTTERY,
    )
    task_id = orchestrator.new_task_id()
    store.create(TaskState(task_id=task_id, status=TaskStatus.queued))
    await orchestrator.run_generation(task_id, req)

    state = store.get(task_id)
    assert state.status == TaskStatus.done
    report = state.outputs.final_diff_report
    assert report.dimension(Dimension.bottom).score == 1.0
    assert report.passed
