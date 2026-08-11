"""相机参数转换：OpenCV(meta_data.json) → Blender。纯 numpy，可脱离 Blender 单测。

meta_data.json 约定（camera_model=OPENCV）：
- camtoworld：4×4 相机到世界位姿，OpenCV 相机系（+X 右, +Y 下, +Z 前，看向 +Z）。
- intrinsics：3×3，fx,fy,cx,cy（像素）。
- 分辨率 width×height。

Blender 相机系：+X 右, +Y 上, −Z 前（看向 −Z）。故 c2w 需右乘 diag(1,-1,-1,1)。
"""
from __future__ import annotations

import numpy as np

# OpenCV 相机系 → Blender 相机系（翻转 Y、Z 轴）
OPENCV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


def c2w_opencv_to_blender(c2w) -> np.ndarray:
    """OpenCV 相机到世界矩阵 → Blender matrix_world。平移不变，仅旋转基变换。"""
    c2w = np.asarray(c2w, dtype=float)
    if c2w.shape != (4, 4):
        raise ValueError("camtoworld 必须是 4×4")
    return c2w @ OPENCV_TO_BLENDER


def intrinsics_to_blender(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int, sensor_width: float = 32.0,
) -> dict:
    """内参 → Blender 相机数据字段。

    返回 dict：lens(mm)、sensor_width/height、sensor_fit、shift_x、shift_y。
    - 光心参考取 (N-1)/2（像素中心约定）：cx=cy=(N-1)/2 时 shift=0。
    - 假定方形像素（fx≈fy）；非方形时以 sensor_fit 主轴的焦距为准。
    """
    sensor_fit = "HORIZONTAL" if width >= height else "VERTICAL"
    if sensor_fit == "HORIZONTAL":
        lens = fx * sensor_width / width
        sensor_height = sensor_width * height / width
    else:
        sensor_height = sensor_width
        lens = fy * sensor_height / height

    base = max(width, height)
    # Blender：shift_x 正向左移画面内容；y 轴向上，故对图像(y 下)取反
    shift_x = ((width - 1) / 2.0 - cx) / base
    shift_y = (cy - (height - 1) / 2.0) / base
    return {
        "lens": float(lens),
        "sensor_width": float(sensor_width),
        "sensor_height": float(sensor_height),
        "sensor_fit": sensor_fit,
        "shift_x": float(shift_x),
        "shift_y": float(shift_y),
    }


def horizontal_fov(fx: float, width: int) -> float:
    """水平视场角(弧度)，便于校验。"""
    return float(2.0 * np.arctan(width / (2.0 * fx)))


def frame_intrinsics(frame: dict) -> tuple[float, float, float, float]:
    """从 meta frame 取 (fx, fy, cx, cy)。"""
    k = frame["intrinsics"]
    return float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])
