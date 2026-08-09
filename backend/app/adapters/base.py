"""Hunyuan3D 适配器接口与数据结构（方案 §3 / §8）。

编排器只依赖这里的抽象接口，具体推理由 Mock 或真实 HTTP 适配器提供。
这样无 GPU 环境用 Mock 跑通全链路，接入真实混元 3D 服务时替换实现即可。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import trimesh

from app.core.preprocess import TextConstraints


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
    """一次生成的产物：几何 + 纹理描述 + 溯源。"""

    mesh: trimesh.Trimesh
    texture: TextureDescriptor
    source: str  # "image" | "text"
    provided_views: set[str] = field(default_factory=set)  # 图通道实际提供的视角


class Hunyuan3DAdapter(abc.ABC):
    """混元 3D 推理适配器抽象接口。"""

    @abc.abstractmethod
    def image_to_3d(
        self, images: list, prompt: str, constraints: TextConstraints
    ) -> GenerationResult:
        """图生 3D（通道 A，主几何）。"""

    @abc.abstractmethod
    def text_to_3d(self, prompt: str, constraints: TextConstraints) -> GenerationResult:
        """文生 3D（通道 B，语义参照）。"""

    @abc.abstractmethod
    def refine_geometry(
        self,
        result: GenerationResult,
        focus: list[str],
        reference: GenerationResult,
        constraints: TextConstraints,
    ) -> GenerationResult:
        """按 focus 维度对几何做定向修正（如底部/比例），可参考 reference。"""

    @abc.abstractmethod
    def repaint(
        self,
        result: GenerationResult,
        focus: list[str],
        constraints: TextConstraints,
    ) -> GenerationResult:
        """按文字约束重绘纹理/材质（Hunyuan3D-Paint，修正 pattern/material）。"""
