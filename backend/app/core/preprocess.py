"""输入预处理与文字约束抽取（方案 §4.1 / §5.1）。

文字约束抽取的目标产物是一个结构化 schema，作为比对引擎的「标准答案」来源之一。

生产环境应替换 ``extract_constraints`` 为 LLM 调用（few-shot 抽取）；此处提供
一个确定性的中文关键词启发式实现，无需外部依赖即可运行与测试。
"""
from __future__ import annotations

import re
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
    # 数值尺寸（统一折算为厘米）。文字独有信息：图像只给得出比例，给不出绝对尺寸。
    height_cm: Optional[float] = None
    width_cm: Optional[float] = None  # 口径 / 直径 / 宽 / 腹径
    raw: str = ""

    def has_numeric_size(self) -> bool:
        """文字是否给出了可用于标定的完整数值尺寸（高 + 口径）。"""
        return self.height_cm is not None and self.width_cm is not None

    def aspect(self) -> Optional[float]:
        """文字规定的高宽比（高 / 口径）。两个数值都给出时才成立。"""
        if self.height_cm and self.width_cm:
            return self.height_cm / self.width_cm
        return None

    def mentions(self, key: str) -> bool:
        return {
            "material": bool(self.materials),
            "pattern": bool(self.patterns),
            "bottom": self.bottom_type is not None,
            "height": self.proportion_hint is not None or self.height_cm is not None,
            "width": (
                self.proportion_hint is not None
                or self.symmetry is not None
                or self.width_cm is not None
            ),
            "shape": self.category is not None or self.proportion_hint is not None,
        }.get(key, False)


# 关键词词典（可扩展 / 可由配置注入）
_MATERIAL_WORDS = {
    "陶瓷": "ceramic", "瓷": "ceramic", "青花": "ceramic",
    "陶器": "ceramic", "陶": "ceramic",
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
_CATEGORY_WORDS = [
    "花瓶", "瓶", "杯", "碗", "壶", "盘", "鼎", "鬲", "簋", "尊", "陶罐", "罐", "陶器",
    "雕像", "摆件", "机器人", "潜艇", "鲸鱼", "玩偶",
]

# --- 数值尺寸抽取 -------------------------------------------------------------
# 形如「高17.5」「口径 16.5 公分」「直径8厘米」「宽 12cm」。单位缺省按厘米折算
# （中文器物描述的惯例，如「高17.5、口径16.5公分」中前者省略了单位）。
_UNIT_TO_CM = {
    "厘米": 1.0, "公分": 1.0, "cm": 1.0, "CM": 1.0,
    "毫米": 0.1, "mm": 0.1, "MM": 0.1,
    "米": 100.0, "m": 100.0, "M": 100.0,
}
# 键名按长度降序，保证「高度」先于「高」、「宽度」先于「宽」匹配
_HEIGHT_KEYS = ["通高", "高度", "高"]
_WIDTH_KEYS = ["最大直径", "腹径", "口径", "底径", "直径", "宽度", "宽", "长度", "长"]
_SIZE_RE = re.compile(
    r"(" + "|".join(_HEIGHT_KEYS + _WIDTH_KEYS) + r")"
    r"\s*(?:约|为|:|：|=)?\s*"
    r"(\d+(?:\.\d+)?)"
    r"\s*(厘米|公分|毫米|cm|CM|mm|MM|米|m|M)?"
)
# 合理尺寸区间（厘米）：过滤掉「公元前711年」这类被误配的大数
_SIZE_MIN_CM, _SIZE_MAX_CM = 0.1, 2000.0


def extract_sizes(text: str) -> tuple[Optional[float], Optional[float]]:
    """抽取 (高度, 宽度/口径)，单位统一为厘米；未提及返回 None。

    同一维度出现多次时取第一次出现的值（描述通常先给主尺寸）。
    """
    height: Optional[float] = None
    width: Optional[float] = None
    for key, num, unit in _SIZE_RE.findall(text):
        cm = float(num) * _UNIT_TO_CM.get(unit, 1.0)
        if not (_SIZE_MIN_CM <= cm <= _SIZE_MAX_CM):
            continue
        if key in _HEIGHT_KEYS and height is None:
            height = cm
        elif key in _WIDTH_KEYS and width is None:
            width = cm
    return height, width


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

    c.height_cm, c.width_cm = extract_sizes(text)

    return c


def validate_images(images) -> None:
    """输入图片校验（数量与必填正图已在 schema 层保证，这里留作扩展点）。

    生产环境应在此接入去背景 / 视角识别 / 相机位姿估计（方案 §4.1）。
    """
    return None
