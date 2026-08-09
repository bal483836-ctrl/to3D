"""Pydantic 数据契约：请求、差异报告、任务状态。

对应方案 §4.4 / §7 / §9。所有对外 API 与内部流转都以这里的模型为准。
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# --- 枚举 --------------------------------------------------------------------


class ViewName(str, Enum):
    """多视图槽位，对应 UI 的 8 个上传位。"""

    front = "front"
    back = "back"
    left = "left"
    right = "right"
    left45 = "left45"
    right45 = "right45"
    top = "top"
    bottom = "bottom"


class FusionStrategy(str, Enum):
    """图文融合主导策略（方案 §5.4 / §6.1）。"""

    image_primary = "image_primary"  # 以图为主（默认）
    balanced = "balanced"  # 图文均衡
    text_correct = "text_correct"  # 以文校正


class TaskStatus(str, Enum):
    """任务状态机（方案 §9）。"""

    queued = "queued"
    preprocessing = "preprocessing"
    image3d = "image3d"
    text3d = "text3d"
    comparing = "comparing"
    refining = "refining"
    awaiting_user = "awaiting_user"
    done = "done"
    failed = "failed"


class Dimension(str, Enum):
    """六大比对维度（方案 §5.2）。"""

    shape = "shape"  # 器型
    pattern = "pattern"  # 花纹
    bottom = "bottom"  # 底部
    height = "height"  # 高度
    width = "width"  # 宽度
    material = "material"  # 材质


DIMENSION_LABELS: dict[Dimension, str] = {
    Dimension.shape: "器型",
    Dimension.pattern: "花纹",
    Dimension.bottom: "底部",
    Dimension.height: "高度",
    Dimension.width: "宽度",
    Dimension.material: "材质",
}

# 各品类默认权重可扩展；此处给出通用默认（器型/材质/花纹权重更高）。
DEFAULT_WEIGHTS: dict[Dimension, float] = {
    Dimension.shape: 0.30,
    Dimension.pattern: 0.15,
    Dimension.bottom: 0.10,
    Dimension.height: 0.10,
    Dimension.width: 0.10,
    Dimension.material: 0.25,
}


# --- 输入 --------------------------------------------------------------------


class ImageInput(BaseModel):
    view: ViewName
    url: str = Field(..., description="图片可访问 URL 或 data URI")
    required: bool = False


class FusionConfig(BaseModel):
    strategy: FusionStrategy = FusionStrategy.image_primary
    consistency_threshold: float = Field(0.8, ge=0.0, le=1.0)
    max_refine_rounds: int = Field(2, ge=0, le=5)
    dimension_weights: Optional[dict[Dimension, float]] = None

    def weights(self) -> dict[Dimension, float]:
        w = dict(DEFAULT_WEIGHTS)
        if self.dimension_weights:
            w.update(self.dimension_weights)
        total = sum(w.values()) or 1.0
        return {k: v / total for k, v in w.items()}


class GenerationRequest(BaseModel):
    images: list[ImageInput] = Field(..., min_length=2, max_length=8)
    prompt: str = Field(..., min_length=1)
    model_version: str = "hunyuan3d-v3.1"
    face_count: str = "1.5m"
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    human_in_loop: bool = False

    @field_validator("images")
    @classmethod
    def _must_include_front(cls, v: list[ImageInput]) -> list[ImageInput]:
        # UI 中「正图」带 * 为必填（方案 §4.1）。
        if not any(img.view == ViewName.front for img in v):
            raise ValueError("必须包含正图 (view=front)")
        return v


# --- 输出 --------------------------------------------------------------------


class DimensionResult(BaseModel):
    dimension: Dimension
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    detail: str = ""
    authority: str = "image"  # image | text | low_confidence，冲突仲裁结果（§5.4）


class DiffReport(BaseModel):
    overall: float
    threshold: float
    passed: bool
    dimensions: list[DimensionResult]
    actions: list[str] = Field(default_factory=list)

    def dimension(self, dim: Dimension) -> DimensionResult:
        for d in self.dimensions:
            if d.dimension == dim:
                return d
        raise KeyError(dim)


class ModelOutputs(BaseModel):
    glb: Optional[str] = None
    obj: Optional[str] = None
    textures: list[str] = Field(default_factory=list)
    final_diff_report: Optional[DiffReport] = None
    provenance: dict = Field(default_factory=dict)


class TaskState(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    round: int = 0
    diff_report: Optional[DiffReport] = None
    preview_url: Optional[str] = None
    outputs: Optional[ModelOutputs] = None
    error: Optional[str] = None


class DecisionAction(str, Enum):
    accept_all = "accept_all"
    reject_keep_A = "reject_keep_A"
    accept_partial = "accept_partial"
    add_prompt = "add_prompt"


class DecisionRequest(BaseModel):
    action: DecisionAction
    dimensions: list[Dimension] = Field(default_factory=list)
    extra_prompt: str = ""
