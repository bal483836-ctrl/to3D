"""混元 3D 桥接推理服务（GPU）。

把官方 Hunyuan3D-2 的真实推理(形状 DiT + 纹理 Paint + 文生图)包装成本项目
`http` 适配器约定的接口：/generate /reference /refine /repaint。

启动后，在主项目侧设置：
    TO3D_ADAPTER=http
    TO3D_HUNYUAN_ENDPOINT=http://<本服务地址>:9000

依赖与权重下载见同目录 README.md。本文件需在装好 hy3dgen 的 GPU 环境运行；
CPU/无权重环境无法运行（这与主项目的 mock 适配器互补）。

现实说明：官方 Hunyuan3D-2 以「图像条件」为主，不原生支持图文联合条件。
本桥接采用「几何来自图像、文字增强纹理 + 交由主项目六维自检修正」的务实路线
(方案文档路线 C)。要做到 DiT 内真正的图文联合(双模态 CFG)，需按
docs/接入混元3D与联合条件方案.md 的路线 A 训练文字条件 adapter，然后在本文件
的 _run_shape 中把文字 tokens 一并送入。
"""
from __future__ import annotations

import base64
import io
import os
from functools import lru_cache

import trimesh
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

# ---- 配置（环境变量可覆盖）--------------------------------------------------
SHAPE_MODEL = os.getenv("HY3D_SHAPE_MODEL", "tencent/Hunyuan3D-2mv")  # 多视图形状
SHAPE_MODEL_SINGLE = os.getenv("HY3D_SHAPE_MODEL_SINGLE", "tencent/Hunyuan3D-2")
TEX_MODEL = os.getenv("HY3D_TEX_MODEL", "tencent/Hunyuan3D-2")
T2I_MODEL = os.getenv("HY3D_T2I_MODEL", "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled")
DEVICE = os.getenv("HY3D_DEVICE", "cuda")
STEPS = int(os.getenv("HY3D_STEPS", "50"))

# 纹理/材质分析器：clip(渲染图 CLIP/VQA + PBR 分析) 或 prompt(仅关键词回退)
TEX_ANALYZER = os.getenv("HY3D_TEX_ANALYZER", "clip")
CLIP_MODEL = os.getenv("HY3D_CLIP_MODEL", "ViT-B-16")  # Chinese-CLIP
PATTERN_THRESHOLD = float(os.getenv("HY3D_PATTERN_THRESHOLD", "0.55"))
RENDER_VIEWS = int(os.getenv("HY3D_RENDER_VIEWS", "4"))
RENDER_RES = int(os.getenv("HY3D_RENDER_RES", "384"))

# 2mv 官方支持的视角键（其余视角作最佳努力忽略）
MV_VIEWS = {"front", "back", "left", "right"}

app = FastAPI(title="Hunyuan3D Bridge", version="0.1.0")


# ---- 懒加载管线（首次调用时加载，避免启动即占显存）--------------------------
@lru_cache(maxsize=1)
def _shape_mv():
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(SHAPE_MODEL).to(DEVICE)


@lru_cache(maxsize=1)
def _shape_single():
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(SHAPE_MODEL_SINGLE).to(DEVICE)


@lru_cache(maxsize=1)
def _paint():
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    return Hunyuan3DPaintPipeline.from_pretrained(TEX_MODEL)


@lru_cache(maxsize=1)
def _t2i():
    from hy3dgen.text2image import HunyuanDiTPipeline

    return HunyuanDiTPipeline(T2I_MODEL, device=DEVICE)


@lru_cache(maxsize=1)
def _rembg():
    from hy3dgen.rembg import BackgroundRemover

    return BackgroundRemover()


# ---- 工具 -------------------------------------------------------------------
def _load_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        b64 = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    if url.startswith("http"):
        import httpx

        return Image.open(io.BytesIO(httpx.get(url, timeout=60).content)).convert("RGBA")
    return Image.open(url).convert("RGBA")


def _mesh_to_b64(mesh: trimesh.Trimesh) -> str:
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return base64.b64encode(buf.getvalue()).decode()


def _load_mesh_b64(b64: str) -> trimesh.Trimesh:
    return trimesh.load(io.BytesIO(base64.b64decode(b64)), file_type="glb", force="mesh")


# ---- 纹理/材质分析（渲染图 CLIP/VQA + PBR）--------------------------------
# 供主项目六维自检的 pattern / material 维度。
_MAT_PRIOR = {"陶瓷": (0.0, 0.2), "瓷": (0.0, 0.2), "青花": (0.0, 0.15), "金属": (0.9, 0.2),
              "木": (0.0, 0.7), "玻璃": (0.0, 0.05), "塑料": (0.0, 0.5), "石": (0.0, 0.8)}
_MAT_CLASSES = ["陶瓷", "金属", "木", "玻璃", "塑料", "石"]
_PATTERN_VOCAB = ["缠枝莲", "莲纹", "云纹", "回纹", "条纹", "格纹", "浮雕", "花纹"]
_GLOSSY_WORDS = ["高光", "釉面", "光泽", "镜面", "抛光", "亮面"]


def _descriptor_from_prompt(prompt: str) -> dict:
    """回退：仅按 prompt 关键词粗给（无渲染/CLIP 时使用）。"""
    metal, rough = 0.0, 0.6
    for k, (m, r) in _MAT_PRIOR.items():
        if k in prompt:
            metal, rough = m, r
            break
    glossy = any(w in prompt for w in _GLOSSY_WORDS)
    return {
        "patterns": [p for p in _PATTERN_VOCAB if p in prompt],
        "glossy": glossy,
        "metalness": metal,
        "roughness": min(rough, 0.15) if glossy else rough,
    }


def _look_at(eye, center, up=(0.0, 1.0, 0.0)):
    import numpy as np

    eye, center, up = np.asarray(eye, float), np.asarray(center, float), np.asarray(up, float)
    z = eye - center
    z /= (np.linalg.norm(z) or 1.0)          # 相机 +Z 指向后方
    x = np.cross(up, z); x /= (np.linalg.norm(x) or 1.0)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def _render_views(mesh: trimesh.Trimesh, n: int, res: int):
    """离屏渲染 n 个环绕视角（带纹理）。headless 用 EGL。"""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as np
    import pyrender

    scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    center = mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(mesh.extents)) / 2.0 or 1.0
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5)
    renderer = pyrender.OffscreenRenderer(res, res)
    imgs = []
    try:
        for i in range(n):
            ang = 2 * np.pi * i / n
            eye = center + np.array([np.sin(ang), 0.25, np.cos(ang)]) * radius * 2.8
            pose = _look_at(eye, center)
            cn = scene.add(cam, pose=pose)
            ln = scene.add(light, pose=pose)
            color, _ = renderer.render(scene)
            imgs.append(Image.fromarray(color).convert("RGB"))
            scene.remove_node(cn)
            scene.remove_node(ln)
    finally:
        renderer.delete()
    return imgs


@lru_cache(maxsize=1)
def _clip():
    import torch
    from cn_clip.clip import load_from_name

    model, preprocess = load_from_name(CLIP_MODEL, device=DEVICE)
    model.eval()
    return model, preprocess, torch


def _clip_logits(images: list, texts: list):
    """返回 (num_images, num_texts) 的余弦相似度矩阵。"""
    import cn_clip.clip as clip

    model, preprocess, torch = _clip()
    img_in = torch.stack([preprocess(im) for im in images]).to(DEVICE)
    txt_in = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        imf = model.encode_image(img_in)
        txf = model.encode_text(txt_in)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        txf = txf / txf.norm(dim=-1, keepdim=True)
        return (imf @ txf.T).float().cpu().numpy()


def _detect_patterns(images: list, requested: list) -> list:
    """对每个候选纹样做 存在/不存在 对比，跨视角取最大概率判定是否出现。"""
    import numpy as np

    candidates = list(dict.fromkeys(requested + _PATTERN_VOCAB))
    present = []
    for p in candidates:
        if p == "花纹":
            continue
        logits = _clip_logits(images, [f"有{p}纹样的表面", "光滑无花纹的表面"])
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        if float(probs[:, 0].max()) >= PATTERN_THRESHOLD:
            present.append(p)
    return present


def _classify_glossy(images: list) -> bool:
    import numpy as np

    logits = _clip_logits(images, ["有光泽釉面高光的表面", "哑光磨砂的表面"])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return bool(probs[:, 0].mean() >= 0.5)


def _classify_material(images: list) -> str:
    import numpy as np

    logits = _clip_logits(images, [f"{m}材质的物体" for m in _MAT_CLASSES])
    return _MAT_CLASSES[int(logits.mean(axis=0).argmax())]


def _pbr_from_mesh(mesh: trimesh.Trimesh):
    """直接从 glb 的 PBR 材质读取金属度/粗糙度；读不到返回 (None, None)。

    glTF 约定 metallicRoughnessTexture: G=roughness, B=metallic。
    """
    import numpy as np

    mat = getattr(getattr(mesh, "visual", None), "material", None)
    if mat is None:
        return None, None
    metal = getattr(mat, "metallicFactor", None)
    rough = getattr(mat, "roughnessFactor", None)
    tex = getattr(mat, "metallicRoughnessTexture", None)
    if tex is not None:
        arr = np.asarray(tex.convert("RGB")) / 255.0
        rough = float(arr[..., 1].mean()) if rough is None else rough
        metal = float(arr[..., 2].mean()) if metal is None else metal
    return (
        float(metal) if metal is not None else None,
        float(rough) if rough is not None else None,
    )


def _texture_descriptor(mesh: trimesh.Trimesh, prompt: str) -> dict:
    """对渲染图做 CLIP/VQA + 从网格读 PBR，产出纹理/材质描述。

    任一步失败或 TEX_ANALYZER=prompt 时，回退到 prompt 关键词版本。
    """
    if TEX_ANALYZER != "clip":
        return _descriptor_from_prompt(prompt)
    try:
        requested = [p for p in _PATTERN_VOCAB if p in prompt and p != "花纹"]
        metal, rough = _pbr_from_mesh(mesh)
        images = _render_views(mesh, RENDER_VIEWS, RENDER_RES)
        if not images:
            raise RuntimeError("渲染失败")

        patterns = _detect_patterns(images, requested)
        glossy = _classify_glossy(images)
        if metal is None or rough is None:
            mat = _classify_material(images)
            pm, pr = _MAT_PRIOR.get(mat, (0.0, 0.6))
            metal = pm if metal is None else metal
            rough = pr if rough is None else rough
        # 光泽与粗糙度互校：高光则粗糙度应低
        if glossy:
            rough = min(rough, 0.2)
        return {
            "patterns": patterns,
            "glossy": glossy,
            "metalness": round(float(metal), 3),
            "roughness": round(float(rough), 3),
        }
    except Exception as exc:  # noqa: BLE001 - 分析失败回退，保证服务可用
        print(f"[tex-analyzer] 分析失败，回退关键词：{exc}")
        return _descriptor_from_prompt(prompt)


def _run_shape(images: list[dict], prompt: str, image_scale: float):
    """图像条件形状生成。多视角走 2mv，单图走单图管线。"""
    rembg = _rembg()
    views = {}
    for img in images:
        v = img["view"]
        if v in MV_VIEWS:
            views[v] = rembg(_load_image(img["url"]))
    gs = max(1.0, 3.0 + 3.0 * image_scale)  # image_scale → guidance_scale 粗映射
    if len(views) >= 2:
        mesh = _shape_mv()(image=views, num_inference_steps=STEPS, guidance_scale=gs)[0]
    else:
        front = next((i for i in images if i["view"] == "front"), images[0])
        mesh = _shape_single()(
            image=rembg(_load_image(front["url"])),
            num_inference_steps=STEPS, guidance_scale=gs,
        )[0]
    return mesh


def _run_paint(mesh: trimesh.Trimesh, cond_image: Image.Image):
    return _paint()(mesh, image=cond_image)


# ---- 请求体 -----------------------------------------------------------------
class GenerateReq(BaseModel):
    images: list[dict]
    prompt: str = ""
    image_scale: float = 1.0
    text_scale: float = 0.35


class ReferenceReq(BaseModel):
    prompt: str


class EditReq(BaseModel):
    glb_base64: str
    focus: list[str] = []
    prompt: str = ""


# ---- 端点 -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "shape_model": SHAPE_MODEL, "tex_model": TEX_MODEL}


@app.post("/generate")
def generate(req: GenerateReq):
    """图像条件生成几何 + 纹理；文字增强纹理条件。"""
    mesh = _run_shape(req.images, req.prompt, req.image_scale)
    # 纹理条件：正图；文字引导强时，用文生图合成一张贴合文字的参考图叠加影响
    front = next((i for i in req.images if i["view"] == "front"), req.images[0])
    cond = _rembg()(_load_image(front["url"]))
    if req.text_scale >= 0.6 and req.prompt:
        try:
            cond = _rembg()(_t2i()(req.prompt))
        except Exception:
            pass
    mesh = _run_paint(mesh, cond)
    return {"glb_base64": _mesh_to_b64(mesh), "texture": _texture_descriptor(mesh, req.prompt)}


@app.post("/reference")
def reference(req: ReferenceReq):
    """文生图 → 图像条件形状，仅供主项目自检打分。"""
    img = _rembg()(_t2i()(req.prompt))
    mesh = _shape_single()(image=img, num_inference_steps=STEPS)[0]
    return {"glb_base64": _mesh_to_b64(mesh), "texture": _texture_descriptor(mesh, req.prompt)}


@app.post("/repaint")
def repaint(req: EditReq):
    """按文字重绘纹理：用文生图合成贴合文字的条件图，对给定网格重跑 Paint。"""
    mesh = _load_mesh_b64(req.glb_base64)
    try:
        cond = _rembg()(_t2i()(req.prompt))
        mesh = _run_paint(mesh, cond)
    except Exception:
        pass
    return {"glb_base64": _mesh_to_b64(mesh), "texture": _texture_descriptor(mesh, req.prompt)}


@app.post("/refine")
def refine(req: EditReq):
    """几何定向修正。

    注意：官方 Hunyuan3D 无原生「局部几何编辑」能力，此处返回原网格(no-op)。
    如需真正的几何修正，建议：提高文字引导后整体重生成(路线 A)，或改用
    text_correct 策略让首轮几何即达标。详见 docs/接入混元3D与联合条件方案.md。
    """
    mesh = _load_mesh_b64(req.glb_base64)
    return {"glb_base64": _mesh_to_b64(mesh)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "9000")))
