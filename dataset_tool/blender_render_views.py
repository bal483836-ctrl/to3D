"""Blender 脚本：按参考 meta_data.json 的 50 组相机参数渲染四种模态数据。

对每个视角输出：Color(彩色) / Depth(深度) / Normal(法线) / Mask(掩码)，
四者由同一相机、同一 800×800 分辨率生成；并为该文物写出结构一致的 meta_data.json。
相机的 camtoworld / intrinsics / 顺序 / 分辨率严格取自参考文件，不随机、不修改。

用法（无头）：
    blender -b --python tools/blender_render_views.py -- \
        --model /path/artifact.glb \
        --meta  datasets/reference_meta_data.json \
        --out   /path/out_dir \
        [--engine CYCLES|BLENDER_EEVEE] [--samples 64] [--normalize-depth]

依赖：Blender 3.x/4.x。本脚本需在 Blender 内运行（import bpy）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import bpy  # 仅在 Blender 内可用
import numpy as np
from mathutils import Matrix

# 让 Blender 的 Python 能找到同目录的 camera_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import camera_utils as cu  # noqa: E402
from meta_utils import build_artifact_meta  # noqa: E402


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="文物模型(glb/gltf/obj/fbx/ply)")
    p.add_argument("--meta", required=True, help="参考 meta_data.json")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--normalize-depth", action="store_true",
                   help="额外输出归一化深度 PNG（默认深度为 EXR 保真）")
    return p.parse_args(argv)


def _clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise ValueError(f"不支持的模型格式: {ext}")
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def _setup_render(scene, width: int, height: int, engine: str, samples: int) -> None:
    scene.render.engine = engine
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True  # 背景透明 → alpha 作掩码
    if engine == "CYCLES":
        scene.cycles.samples = samples
    else:
        scene.eevee.taa_render_samples = samples
    # 打开需要的渲染通道
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_normal = True
    # 数据通道用 Raw，避免 sRGB 影响
    scene.view_settings.view_transform = "Standard"


def _setup_compositor(scene, out_dir: str, normalize_depth: bool):
    """搭建合成节点：把 Color/Depth/Normal/Mask 分别写文件。返回可更新的 File Output 节点。"""
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")

    def file_out(name, subtype, color_depth="8", fmt="PNG"):
        n = tree.nodes.new("CompositorNodeOutputFile")
        n.label = name
        n.base_path = out_dir
        n.format.file_format = fmt
        n.format.color_depth = color_depth
        if fmt == "OPEN_EXR":
            n.format.color_mode = "RGB"
        return n

    # Color：直接用渲染 Image
    color_out = file_out("color", "color")
    color_out.format.color_mode = "RGBA"
    tree.links.new(rl.outputs["Image"], color_out.inputs[0])

    # Mask：用 Alpha（film 透明背景 → 物体处 alpha=1）
    mask_out = file_out("mask", "mask")
    mask_out.format.color_mode = "BW"
    tree.links.new(rl.outputs["Alpha"], mask_out.inputs[0])

    # Normal：[-1,1] → [0,1]  (N*0.5+0.5)
    scale = tree.nodes.new("CompositorNodeMixRGB")
    scale.blend_type = "MULTIPLY"
    scale.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    tree.links.new(rl.outputs["Normal"], scale.inputs[1])
    bias = tree.nodes.new("CompositorNodeMixRGB")
    bias.blend_type = "ADD"
    bias.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    tree.links.new(scale.outputs[0], bias.inputs[1])
    normal_out = file_out("normal", "normal")
    normal_out.format.color_mode = "RGB"
    tree.links.new(bias.outputs[0], normal_out.inputs[0])

    # Depth：EXR 保真
    depth_out = file_out("depth", "depth", color_depth="32", fmt="OPEN_EXR")
    tree.links.new(rl.outputs["Depth"], depth_out.inputs[0])

    outs = {"color": color_out, "mask": mask_out, "normal": normal_out, "depth": depth_out}

    depth_png = None
    if normalize_depth:
        # 归一化到 near..far → 0..1（scene_box.near/far）
        nb = scene.get("_near", 2.0)
        fb = scene.get("_far", 6.0)
        norm = tree.nodes.new("CompositorNodeMapRange")
        norm.inputs[1].default_value = nb  # From Min
        norm.inputs[2].default_value = fb  # From Max
        norm.inputs[3].default_value = 0.0
        norm.inputs[4].default_value = 1.0
        tree.links.new(rl.outputs["Depth"], norm.inputs[0])
        depth_png = file_out("depth_png", "depth")
        depth_png.format.color_mode = "BW"
        tree.links.new(norm.outputs[0], depth_png.inputs[0])
        outs["depth_png"] = depth_png
    return outs


def _apply_camera(cam_obj, cam_data, frame: dict, width: int, height: int) -> None:
    # 外参：OpenCV c2w → Blender matrix_world
    m = cu.c2w_opencv_to_blender(frame["camtoworld"])
    cam_obj.matrix_world = Matrix([list(r) for r in m])
    # 内参
    fx, fy, cx, cy = cu.frame_intrinsics(frame)
    b = cu.intrinsics_to_blender(fx, fy, cx, cy, width, height)
    cam_data.type = "PERSP"
    cam_data.sensor_fit = b["sensor_fit"]
    cam_data.sensor_width = b["sensor_width"]
    cam_data.sensor_height = b["sensor_height"]
    cam_data.lens = b["lens"]
    cam_data.shift_x = b["shift_x"]
    cam_data.shift_y = b["shift_y"]


def _rename_frame_outputs(out_dir: str, idx: int, outs: dict) -> None:
    """File Output 会写成 <label>0001.png 之类，重命名为 {idx}_{modality}.ext。"""
    mapping = {
        "color": (f"{idx}_colors.png", ".png"),
        "depth": (f"{idx}_depth.exr", ".exr"),
        "normal": (f"{idx}_normal.png", ".png"),
        "mask": (f"{idx}_mask.png", ".png"),
        "depth_png": (f"{idx}_depth.png", ".png"),
    }
    for key, node in outs.items():
        target_name, ext = mapping[key]
        # 找到该节点本轮写出的文件（以 label 开头、对应扩展名）
        prefix = node.label
        for fn in os.listdir(out_dir):
            if fn.startswith(prefix) and fn.endswith(ext):
                src = os.path.join(out_dir, fn)
                dst = os.path.join(out_dir, target_name)
                if src != dst:
                    os.replace(src, dst)
                break


def _write_meta(meta: dict, out_dir: str) -> None:
    """写出与参考结构一致的 meta_data.json，补充各模态路径。"""
    out = build_artifact_meta(meta)
    with open(os.path.join(out_dir, "meta_data.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=4, ensure_ascii=False)


def main() -> None:
    args = _parse_args()
    meta = json.load(open(args.meta, encoding="utf-8"))
    width, height = meta["width"], meta["height"]
    os.makedirs(args.out, exist_ok=True)

    _clear_scene()
    _import_model(args.model)
    scene = bpy.context.scene
    scene["_near"] = meta.get("scene_box", {}).get("near", 2.0)
    scene["_far"] = meta.get("scene_box", {}).get("far", 6.0)

    # 中性照明（不影响 Normal/Depth/Mask；Color 需要光）
    world = bpy.data.worlds.new("W") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

    cam_data = bpy.data.cameras.new("Cam")
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    _setup_render(scene, width, height, args.engine, args.samples)
    outs = _setup_compositor(scene, args.out, args.normalize_depth)

    for i, frame in enumerate(meta["frames"]):
        _apply_camera(cam_obj, cam_data, frame, width, height)
        bpy.ops.render.render(write_still=False)  # 由 File Output 节点写文件
        _rename_frame_outputs(args.out, i, outs)
        print(f"[render] view {i}/{len(meta['frames'])} done", flush=True)

    _write_meta(meta, args.out)
    print(f"[done] 共 {len(meta['frames'])} 视角 × 4 模态 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
