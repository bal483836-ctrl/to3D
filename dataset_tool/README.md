# 50 视角数据集生成工具（独立 · 上传 3D 模型）

独立的小工具：**上传一个 3D 模型**，按参考 `reference_meta_data.json` 的 **50 组固定
相机参数**渲染 **Color / Depth / Normal / Mask** 四模态数据，并输出结构一致的
`meta_data.json`，最终打包 zip 下载。与主产品完全解耦，适合本地 / VSCode 单独运行。

**严格约束**：所有文物使用参考文件里**相同**的 50 组 `camtoworld`、`intrinsics`、
视角顺序、`800×800`；脚本**不随机、不修改**相机参数。

---

## 在 VSCode 中运行
1. 打开项目根目录，安装 Python 扩展。
2. 终端（或 VSCode 任务）安装依赖：
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r dataset_tool/requirements.txt
   ```
3. 安装 **Blender 3.x/4.x**，让工具能找到它：
   ```bash
   export TO3D_BLENDER_BIN=/path/to/blender     # 或把 blender 加进 PATH
   ```
4. 按 **F5** 选择 **“运行：50 视角数据集工具（上传3D模型）”**（见 `.vscode/launch.json`），
   或命令行：
   ```bash
   cd dataset_tool && python app.py         # 或 uvicorn app:app --port 8100
   ```
5. 打开 http://127.0.0.1:8100 → 拖入/选择 3D 模型（可选再上传自定义 `meta_data.json`，
   不传则用内置 50 视角参考）→ 点「生成数据集」→ 下载 zip。

也可纯命令行（不经网页）直接渲染：
```bash
blender -b --python dataset_tool/blender_render_views.py -- \
  --model 你的模型.glb --meta dataset_tool/reference_meta_data.json --out out_dir \
  --engine CYCLES --samples 64
```

## 输出
```
out_dir/
  0_colors.png 0_depth.exr 0_normal.png 0_mask.png
  ... （共 50 组，四模态同相机、同 800×800）
  meta_data.json          # 与参考同结构，补 depth/normal/mask 路径，相机参数原样
```

## 配置（环境变量）
| 变量 | 默认 | 说明 |
|------|------|------|
| `TO3D_BLENDER_BIN` | `blender` | Blender 可执行文件路径 |
| `TO3D_DATASET_ENGINE` | `CYCLES` | `CYCLES`（更准）或 `BLENDER_EEVEE`（更快） |
| `TO3D_DATASET_SAMPLES` | `64` | 采样数 |
| `TO3D_DATASET_TIMEOUT` | `3600` | 单次渲染超时(秒) |
| `TO3D_DS_WORKDIR` | `dataset_tool/_work` | 上传/输出工作目录 |
| `PORT` | `8100` | 服务端口 |

## 支持的模型格式
`.glb / .gltf / .obj / .fbx / .ply`

## 测试
```bash
cd dataset_tool && pytest -q      # 相机换算 / meta 结构 / 上传校验 / Blender 缺失处理
```

## 相机参数映射（camera_utils.py，已单测）
- 外参 OpenCV(+X右/+Y下/+Z前) → Blender(+X右/+Y上/−Z前)：右乘 `diag(1,-1,-1,1)`。
- 内参 `fx→lens=fx·sensor/W`；`cx,cy→shift`（以 `(N-1)/2` 为光心，参考数据 → shift=0 居中）。

## 说明
- 渲染由系统 Blender 完成；无 Blender 时网页会给出明确提示。
- 若下游需要**相机系法线**而非世界系，在 `blender_render_views.py` 的合成器里对
  Normal 加一次视图矩阵变换。
