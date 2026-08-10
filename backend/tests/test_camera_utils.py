"""相机参数转换单测（脱离 Blender）。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# 让测试能 import tools/camera_utils
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
import camera_utils as cu  # noqa: E402

REF = os.path.join(_ROOT, "datasets", "reference_meta_data.json")


def test_opencv_to_blender_flips_y_z():
    c2w = np.eye(4)
    b = cu.c2w_opencv_to_blender(c2w)
    # 相机 +Y、+Z 轴取反，+X 不变
    assert np.allclose(b[:3, 0], [1, 0, 0])
    assert np.allclose(b[:3, 1], [0, -1, 0])
    assert np.allclose(b[:3, 2], [0, 0, -1])
    # 平移不变
    c2w2 = np.eye(4); c2w2[:3, 3] = [1, 2, 3]
    assert np.allclose(cu.c2w_opencv_to_blender(c2w2)[:3, 3], [1, 2, 3])


def test_intrinsics_centered_zero_shift():
    b = cu.intrinsics_to_blender(1111.111, 1111.111, 399.5, 399.5, 800, 800)
    assert b["sensor_fit"] == "HORIZONTAL"
    assert abs(b["shift_x"]) < 1e-9 and abs(b["shift_y"]) < 1e-9
    # lens = fx * sensor_width / W = 1111.111*32/800 ≈ 44.44
    assert abs(b["lens"] - 44.4444) < 1e-2


def test_intrinsics_offcenter_shift_sign():
    # cx 右移(变大) → 画面内容左移 → shift_x 为负
    b = cu.intrinsics_to_blender(1111.0, 1111.0, 500.0, 399.5, 800, 800)
    assert b["shift_x"] < 0


def test_fov_matches_reference():
    fov = cu.horizontal_fov(1111.1113654242622, 800)
    assert abs(np.degrees(fov) - 39.6) < 0.5


def test_reference_meta_50_frames():
    ref = json.load(open(REF, encoding="utf-8"))
    assert len(ref["frames"]) == 50
    assert ref["width"] == 800 and ref["height"] == 800
    assert ref["camera_model"] == "OPENCV"
