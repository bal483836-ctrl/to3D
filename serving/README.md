# 混元 3D 桥接推理服务

把官方 **Hunyuan3D-2** 的真实推理包装成主项目 `http` 适配器约定的接口，让
`to3D` 用上真实模型（而非 mock）。

> 现实提醒：官方 Hunyuan3D-2 以**图像条件**为主，不原生支持「图文联合(双模态)」。
> 本桥接走务实路线：**几何来自图像、文字增强纹理，再交给主项目六维自检修正**。
> 要在 DiT 内做真正的图文联合(双模态 CFG)，需训练文字条件 adapter，见
> [`../docs/接入混元3D与联合条件方案.md`](../docs/接入混元3D与联合条件方案.md) 路线 A。

---

## 1. 硬件与系统
- NVIDIA GPU：形状生成建议 ≥ 12GB 显存；开纹理(Paint) 建议 ≥ 16–24GB。
- CUDA 11.8/12.1、Python 3.10/3.11、Linux。

## 2. 安装混元 3D（hy3dgen）
```bash
# 2.1 建环境
conda create -n hunyuan3d python=3.11 -y && conda activate hunyuan3d

# 2.2 安装 PyTorch（按你的 CUDA 版本，示例 cu121）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2.3 拉官方仓库并安装
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2
cd Hunyuan3D-2
pip install -r requirements.txt
pip install -e .
# 纹理烘焙需要的自定义算子（要纹理就装）
cd hy3dgen/texgen/custom_rasterizer && pip install -e . && cd -
cd hy3dgen/texgen/differentiable_renderer && pip install -e . && cd -

# 2.4 本桥接服务额外依赖
pip install fastapi "uvicorn[standard]" trimesh pillow httpx
```

## 3. 下载权重
首次运行管线会自动从 HuggingFace 拉取；也可预先下载：
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download tencent/Hunyuan3D-2mv     # 多视图形状(默认)
huggingface-cli download tencent/Hunyuan3D-2       # 单图形状 + 纹理
# 文生图(用于 /reference 与文字增强纹理)：
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled
```
> 国内网络可设 `export HF_ENDPOINT=https://hf-mirror.com` 加速。
> 显存紧张可改用 mini/turbo 变体：`export HY3D_SHAPE_MODEL_SINGLE=tencent/Hunyuan3D-2mini`。

## 4. 启动桥接服务
```bash
# 在 to3D/serving 目录下（确保 hy3dgen 可 import）
python hunyuan_server.py            # 监听 0.0.0.0:9000
# 自检：
curl http://127.0.0.1:9000/health
```
可用环境变量：`HY3D_SHAPE_MODEL`、`HY3D_SHAPE_MODEL_SINGLE`、`HY3D_TEX_MODEL`、
`HY3D_T2I_MODEL`、`HY3D_DEVICE`(默认 cuda)、`HY3D_STEPS`(默认 50)、`PORT`(默认 9000)。

## 5. 让主项目用上它
在主项目(to3D 根目录)：
```bash
export TO3D_ADAPTER=http
export TO3D_HUNYUAN_ENDPOINT=http://<GPU机器IP>:9000
# 本地起主服务
make dev            # 或 ./scripts/dev.sh ；或 docker compose（见下）
```
`/api/health` 应显示 `"adapter":"http"`。然后在前端正常上传图 + 文字生成即可，
产物为真实混元 3D 模型；六维自检与修正照常工作。

Docker 方式让容器内主服务连到宿主 GPU 服务：
```bash
TO3D_ADAPTER=http TO3D_HUNYUAN_ENDPOINT=http://host.docker.internal:9000 \
  docker compose up --build
```

## 6. 端点契约（主项目↔桥接）
| 端点 | 入参 | 出参 |
|------|------|------|
| `POST /generate`  | `{images, prompt, image_scale, text_scale}` | `{glb_base64, texture{...}}` |
| `POST /reference` | `{prompt}` | `{glb_base64, texture{...}}` |
| `POST /repaint`   | `{glb_base64, focus, prompt}` | `{glb_base64, texture{...}}` |
| `POST /refine`    | `{glb_base64, focus, prompt}` | `{glb_base64}` |

## 7. 纹理/材质分析（让 花纹 & 材质 两维真正生效）
`/generate` `/reference` `/repaint` 返回的 `texture` 由 **真实分析** 得到，而非猜 prompt：
1. **多视角离屏渲染**（pyrender，headless 用 EGL）把带纹理网格渲染成若干张图。
2. **Chinese-CLIP** 对渲染图做零样本判定：
   - 花纹存在性：对每个候选纹样（缠枝莲/云纹/回纹…）做「有该纹样 vs 光滑无花纹」对比，
     跨视角取最大概率，超过阈值即判定出现。
   - 光泽：「釉面高光 vs 哑光磨砂」判定 `glossy`。
   - 材质类别：陶瓷/金属/木/玻璃/塑料/石，用于兜底 PBR 先验。
3. **PBR 读取**：直接从 glb 材质读 `metallicFactor/roughnessFactor` 或
   `metallicRoughnessTexture`(G=roughness,B=metallic) 求均值，最准确。

相关依赖：`pyrender cn_clip numpy`（见 requirements.txt）。headless 渲染需系统库：
```bash
apt-get install -y libegl1 libgl1
```
可调环境变量：`HY3D_TEX_ANALYZER`(clip/prompt)、`HY3D_CLIP_MODEL`(默认 ViT-B-16)、
`HY3D_PATTERN_THRESHOLD`(默认 0.55)、`HY3D_RENDER_VIEWS`(默认 4)、`HY3D_RENDER_RES`(默认 384)。
分析任一步失败会自动回退到关键词版本，服务不中断。设 `HY3D_TEX_ANALYZER=prompt`
可完全关闭重分析（免装 pyrender/cn_clip）。

## 8. 其它局限
- **几何以图为准**：多视图越全（尤其 front/back/left/right），几何越准；缺视角时
  提高文字引导或补图。2mv 主要支持这四个正交视角，45°/顶/底会被忽略作最佳努力。
- **/refine 为 no-op**：官方无局部几何编辑。若几何需贴合文字，建议用
  `text_correct` 策略让首轮即达标，或按路线 A 训练后整体重生成。
