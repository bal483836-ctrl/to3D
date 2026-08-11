"""按参考结构生成单文物 meta_data.json（相机参数原样，补四模态路径）。纯函数，可单测。"""
from __future__ import annotations


def build_artifact_meta(ref: dict) -> dict:
    out = {
        "camera_model": ref["camera_model"],
        "height": ref["height"],
        "width": ref["width"],
        "worldtogt": ref["worldtogt"],
        "scene_box": ref["scene_box"],
        "frames": [],
    }
    for i, f in enumerate(ref["frames"]):
        out["frames"].append({
            "rgb_path": f"{i}_colors.png",
            "depth_path": f"{i}_depth.exr",
            "normal_path": f"{i}_normal.png",
            "mask_path": f"{i}_mask.png",
            "camtoworld": f["camtoworld"],   # 原样，不修改
            "intrinsics": f["intrinsics"],   # 原样，不修改
        })
    return out
