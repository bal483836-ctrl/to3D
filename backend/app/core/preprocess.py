"""输入预处理与文字约束抽取（方案 §4.1 / §5.1）。

文字约束抽取的目标产物是一个结构化 schema，作为比对引擎的「标准答案」来源之一。

生产环境应替换 ``extract_constraints`` 为 LLM 调用（few-shot 抽取）；此处提供
一个确定性的中文关键词启发式实现，无需外部依赖即可运行与测试。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextConstraints:
    """从 prompt 抽取的结构化约束。字段为 None 表示文字未提及该维度。"""

    category: Optional[str] = None
    materials: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    bottom_type: Optional[str] = None
    # 比例意图：细长 → tall，矮胖 → wide，None 表示未提及
    proportion_hint: Optional[str] = None
    symmetry: Optional[bool] = None
    glossy: Optional[bool] = None  # 材质光泽意图：高光/釉面 → True，哑光 → False
    raw: str = ""

    def mentions(self, key: str) -> bool:
        return {
            "material": bool(self.materials),
            "pattern": bool(self.patterns),
            "bottom": self.bottom_type is not None,
            "height": self.proportion_hint is not None,
            "width": self.proportion_hint is not None or self.symmetry is not None,
            "shape": self.category is not None or self.proportion_hint is not None,
        }.get(key, False)


# 关键词词典（可扩展 / 可由配置注入）
_MATERIAL_WORDS = {
    "陶瓷": "ceramic", "瓷": "ceramic", "青花": "ceramic",
    "金属": "metal", "不锈钢": "metal", "铜": "metal", "铁": "metal",
    "木": "wood", "木质": "wood",
    "玻璃": "glass", "塑料": "plastic", "石": "stone", "大理石": "stone",
}
_PATTERN_WORDS = ["缠枝莲", "莲纹", "花纹", "纹样", "云纹", "回纹", "条纹", "格纹", "浮雕"]
_BOTTOM_WORDS = {
    "圈足": "ring_foot", "平底": "flat", "三足": "tripod", "圆底": "round",
    "底座": "base", "尖底": "pointed",
}
_TALL_WORDS = ["细长", "修长", "高挑", "瘦高", "细高"]
_WIDE_WORDS = ["矮胖", "宽扁", "圆润", "敦实", "低矮"]
_GLOSSY_WORDS = ["高光", "釉面", "光泽", "镜面", "抛光", "亮面"]
_MATTE_WORDS = ["哑光", "磨砂", "亚光", "陶土"]
_SYMMETRIC_WORDS = ["对称", "左右对称"]
_CATEGORY_WORDS = ["花瓶", "瓶", "杯", "碗", "壶", "盘", "雕像", "摆件", "机器人", "潜艇", "鲸鱼", "玩偶"]


def extract_constraints(prompt: str) -> TextConstraints:
    """把中文 prompt 解析为结构化约束。确定性、无副作用。"""
    text = prompt.strip()
    c = TextConstraints(raw=text)

    for word in _CATEGORY_WORDS:
        if word in text:
            c.category = word
            break

    for word, canon in _MATERIAL_WORDS.items():
        if word in text and canon not in c.materials:
            c.materials.append(canon)

    for word in _PATTERN_WORDS:
        if word in text and word not in c.patterns:
            c.patterns.append(word)

    for word, canon in _BOTTOM_WORDS.items():
        if word in text:
            c.bottom_type = canon
            break

    if any(w in text for w in _TALL_WORDS):
        c.proportion_hint = "tall"
    elif any(w in text for w in _WIDE_WORDS):
        c.proportion_hint = "wide"

    if any(w in text for w in _GLOSSY_WORDS):
        c.glossy = True
    elif any(w in text for w in _MATTE_WORDS):
        c.glossy = False

    if any(w in text for w in _SYMMETRIC_WORDS):
        c.symmetry = True

    return c


def validate_images(images) -> None:
    """输入图片校验（数量与必填正图已在 schema 层保证，这里留作扩展点）。

    生产环境应在此接入去背景 / 视角识别 / 相机位姿估计（方案 §4.1）。
    """
    return None
