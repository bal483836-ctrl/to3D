"""Hunyuan3D 适配器接口与数据结构（方案 §3 / §8）。

编排器只依赖这里的抽象接口，具体推理由 Mock 或真实 HTTP 适配器提供。
这样无 GPU 环境用 Mock 跑通全链路，接入真实混元 3D 服务时替换实现即可。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import trimesh

from app.core.preprocess import TextConstraints
from app.models.schemas import Guidance


@dataclass
class TextureDescriptor:
    """纹理/材质的语义描述。

    生产环境由对返回贴图做 CLIP/VQA(花纹存在性) 与 PBR 统计(金属度/粗糙度) 得到；
    Mock 中直接构造。比对引擎的 pattern/material 维度基于它评分（方案 §5.2）。
    """

    patterns: list[str] = field(default_factory=list)  # 检出的表面纹样
    glossy: bool = False  # 是否高光/釉面
    metalness: float = 0.0  # 0~1
    roughness: float = 1.0  # 0~1


@dataclass
class GenerationResult:
    """一份联合生成的产物：几何 + 纹理描述 + 溯源。"""

    mesh: trimesh.Trimesh
    texture: TextureDescriptor
    source: str = "joint"  # 联合生成
    provided_views: set[str] = field(default_factory=set)  # 实际提供的视角
    guidance: Guidance | None = None  # 本次生成使用的双模态引导权重


class Hunyuan3DAdapter(abc.ABC):
    """混元 3D 推理适配器抽象接口（图文联合条件生成）。"""

    @abc.abstractmethod
    def generate(
        self,
        images: list,
        prompt: str,
        constraints: TextConstraints,
        guidance: Guidance,
    ) -> GenerationResult:
        """图文联合条件生成：多视图图像特征与文字特征融合为同一条件序列，
        经 CFG 双模态引导，**单次**生成一份模型（几何 + 纹理）。

        guidance.image_scale / text_scale 决定两种模态在联合生成中的相对影响，
        两者同时生效，而非切换到单一模态。
        """

    @abc.abstractmethod
    def build_reference(self, constraints: TextConstraints) -> GenerationResult:
        """仅供一致性自检使用的「规格重建」参照（不作为交付物）。

        由文字约束重建目标规格，用于度量联合产物在各维度上的偏离；
        生产环境中几何自检更应直接用输入图像做多视角轮廓/CLIP 比对。
        """

    @abc.abstractmethod
    def refine_geometry(
        self,
        result: GenerationResult,
        focus: list[str],
        reference: GenerationResult,
        constraints: TextConstraints,
    ) -> GenerationResult:
        """对联合产物按 focus 维度做几何定向修正（提高该维度的文字引导后局部重生成）。"""

    @abc.abstractmethod
    def repaint(
        self,
        result: GenerationResult,
        focus: list[str],
        constraints: TextConstraints,
    ) -> GenerationResult:
        """对联合产物按文字约束重绘纹理/材质（Hunyuan3D-Paint，修正 pattern/material）。"""
