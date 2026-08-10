"""任务编排：图文联合生成 → 一致性自检 → 定向修正闭环（方案 §4 / §9）。

图与文同时作为条件，单次联合生成一份模型；随后对这一份产物做六维自检，
不达标则调高对应维度的文字引导并局部重生成/重绘，有界迭代收敛。
无 GPU 时用 MockHunyuan3DAdapter，全链路可跑、可收敛。
"""
from __future__ import annotations

import asyncio
import os
import uuid

import trimesh

from app.adapters.base import GenerationResult, Hunyuan3DAdapter
from app.adapters.mock import MockHunyuan3DAdapter
from app.config import settings
from app.core import preprocess
from app.core.comparison import compare
from app.observability import logger, track_generation
from app.models.schemas import (
    DecisionAction,
    Dimension,
    GenerationRequest,
    ModelOutputs,
    TaskState,
    TaskStatus,
)
from app.store import store

# 修正动作 → 维度归类
_GEOMETRY_ACTIONS = {"refine_bottom", "refine_shape", "refine_proportion"}
_TEXTURE_ACTIONS = {"repaint_pattern", "adjust_material"}
_ACTION_FOCUS = {
    "repaint_pattern": "pattern",
    "adjust_material": "material",
    "refine_bottom": "bottom",
    "refine_shape": "shape",
    "refine_proportion": "height",
}

_OUTPUT_DIR = settings.output_dir


def get_adapter() -> Hunyuan3DAdapter:
    """选择适配器。设置 TO3D_ADAPTER=http 可接入真实混元 3D 服务。"""
    if settings.adapter == "http":
        from app.adapters.http import HttpHunyuan3DAdapter

        if not settings.hunyuan_endpoint:
            raise RuntimeError("TO3D_ADAPTER=http 需同时设置 TO3D_HUNYUAN_ENDPOINT")
        return HttpHunyuan3DAdapter(settings.hunyuan_endpoint, settings.request_timeout)
    return MockHunyuan3DAdapter()


def _export(mesh: trimesh.Trimesh, task_id: str, tag: str) -> dict:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    base = os.path.join(_OUTPUT_DIR, f"{task_id}_{tag}")
    glb, obj = base + ".glb", base + ".obj"
    mesh.export(glb)
    mesh.export(obj)
    return {"glb": glb, "obj": obj}


def new_task_id() -> str:
    return "gen_" + uuid.uuid4().hex[:12]


# 生成任务并发上限（GPU 通常 1~2）。绑定到运行中的事件循环，懒创建。
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))
    return _semaphore


async def _call(func, *args):
    """把阻塞的适配器调用放线程池执行，带失败重试（网络抖动等）。"""
    loop = asyncio.get_running_loop()
    last: Exception | None = None
    for attempt in range(settings.adapter_retries + 1):
        try:
            return await loop.run_in_executor(None, func, *args)
        except Exception as e:  # noqa: BLE001 - 交由上层统一处理，先重试
            last = e
            if attempt < settings.adapter_retries:
                await asyncio.sleep(2 ** attempt)
                logger.warning("适配器调用失败重试 %d/%d: %s",
                               attempt + 1, settings.adapter_retries, e)
    raise last  # type: ignore[misc]


async def run_generation(task_id: str, req: GenerationRequest) -> None:
    """执行完整生成流程。受并发上限约束；异常时置为 failed。"""
    adapter = get_adapter()
    state = store.get(task_id)
    assert state is not None
    # 超出并发上限时在此等待，期间状态保持 queued
    async with _get_semaphore():
        with track_generation():
            await _run_generation_inner(task_id, req, adapter, state)


async def _run_generation_inner(task_id, req, adapter, state) -> None:
    try:
        # 阶段 0：预处理 + 文字约束抽取
        state.status = TaskStatus.preprocessing
        state.progress = 0.05
        await store.update(state)
        constraints = preprocess.extract_constraints(req.prompt)
        preprocess.validate_images(req.images)

        # 阶段 1：图文联合条件生成（单次，一份模型）
        state.status = TaskStatus.generating
        state.progress = 0.2
        await store.update(state)
        guidance = req.fusion.guidance()
        output = await _call(adapter.generate, req.images, req.prompt, constraints, guidance)
        # 规格参照仅用于自检打分（不交付）
        reference = await _call(adapter.build_reference, constraints)

        # 阶段 2/3：一致性自检 + 定向修正闭环（记录历史最优防发散——方案 §11）
        best: GenerationResult = output
        best_report = None
        for rnd in range(req.fusion.max_refine_rounds + 1):
            state.status = TaskStatus.verifying
            state.round = rnd
            state.progress = min(0.55 + rnd * 0.15, 0.95)
            report = await _call(compare, output, reference, constraints, req.fusion)
            state.diff_report = report
            await store.update(state)

            if best_report is None or report.overall >= best_report.overall:
                best, best_report = output, report

            if report.passed or not report.actions or rnd >= req.fusion.max_refine_rounds:
                break

            # 人在环：暂停等待用户决策（方案 §4.5 / §7.3）
            focus_actions = report.actions
            if req.human_in_loop:
                state.status = TaskStatus.awaiting_user
                await store.update(state)
                decision = await store.wait_decision(task_id)
                if decision.action == DecisionAction.reject_keep_A:
                    break
                focus_actions = _apply_decision(decision, report.actions)

            # 定向修正：提高对应维度的文字引导后局部重生成 / 重绘
            state.status = TaskStatus.refining
            await store.update(state)
            output = await _refine(
                adapter, output, reference, constraints, focus_actions
            )

        # 输出历史最优结果
        state.diff_report = best_report
        files = _export(best.mesh, task_id, "final")
        state.status = TaskStatus.done
        state.progress = 1.0
        state.preview_url = f"/api/v1/generation/{task_id}/mesh"
        state.outputs = ModelOutputs(
            glb=files["glb"],
            obj=files["obj"],
            textures=[],
            final_diff_report=best_report,
            provenance={
                "mode": "joint_image_text",
                "rounds": state.round,
                "strategy": req.fusion.strategy.value,
                "guidance": guidance.model_dump(),
                "prompt": req.prompt,
                "constraints": constraints.__dict__,
            },
        )
        await store.update(state)
    except Exception as exc:  # noqa: BLE001 - 顶层兜底，写入错误状态
        logger.exception("生成任务失败 task=%s", task_id)
        state.status = TaskStatus.failed
        state.error = f"{type(exc).__name__}: {exc}"
        await store.update(state)
        raise


def _apply_decision(decision, actions: list[str]) -> list[str]:
    if decision.action == DecisionAction.accept_all:
        return actions
    if decision.action == DecisionAction.accept_partial:
        keep = {d.value for d in decision.dimensions}
        return [a for a in actions if _ACTION_FOCUS.get(a) in keep]
    return actions


async def _refine(
    adapter: Hunyuan3DAdapter,
    image_result: GenerationResult,
    text_result: GenerationResult,
    constraints,
    actions: list[str],
) -> GenerationResult:
    geom_focus = [_ACTION_FOCUS[a] for a in actions if a in _GEOMETRY_ACTIONS]
    tex_focus = [_ACTION_FOCUS[a] for a in actions if a in _TEXTURE_ACTIONS]
    result = image_result
    if geom_focus:
        result = await _call(adapter.refine_geometry, result, geom_focus, text_result, constraints)
    if tex_focus:
        result = await _call(adapter.repaint, result, tex_focus, constraints)
    return result
