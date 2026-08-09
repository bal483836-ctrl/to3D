# to3D · 图文联合 3D 生成

基于混元 3D（Hunyuan3D）的 **多图 + 文字联合** 3D 生成系统：一份 3D 模型由
多视图图片与文字描述**同时（联合）**驱动 —— 图像特征与文字特征融合成同一个条件，
**一次生成一份模型**，而非"图生一个、文生一个再比较"。核心机制是
**图文联合生成 → 六维一致性自检 → 定向修正闭环**。

- 方案设计（完整）：[`docs/方案设计.md`](docs/方案设计.md)
- 可视化方案页：[`docs/方案设计.html`](docs/方案设计.html)

## 它是怎么工作的

```
多视图(2~8) 图像tokens ┐
                      ⊕ 同一条件序列, CFG 双模态引导 → 单次联合生成 → 一份模型
文字 文本tokens        ┘                                        │
                                                               ▼
                                      六维一致性自检 → 达标? ─是→ 输出
                                                        └否→ 定向修正(调高文字引导) ↺
```

1. **图文联合生成**：多视图图像 tokens（DINO/CLIP）与文字 tokens（CLIP text）拼接为
   同一条件序列，经 CFG 双模态引导（`image_scale`/`text_scale`），Hunyuan3D-DiT + Paint
   **单次**生成一份网格 + 纹理。**这是唯一交付主体**。
2. **一致性自检**：对**这一份产物**在 **器型 / 花纹 / 底部 / 高度 / 宽度 / 材质** 六维打分。
   - 几何维度直接在网格上真实测量（Chamfer/IoU、包围盒比例、底部形态、对称性）。
   - 纹理/材质维度基于纹理描述与文字约束评分（生产环境由 CLIP/VQA/PBR 分析得到）。
   - 参照来自输入图像证据与文字约束规格，**仅用于打分，不作为第二个交付物**。
3. **冲突仲裁**：某维度有对应视角图 → 以图像为准（接受）；无图但文字明确 → 提高文字引导修正。
4. **修正闭环**：调高对应维度文字引导后局部重生成 / 重绘，有界迭代收敛（防发散，取历史最优）。

## 目录结构

```
backend/
  app/
    main.py              # FastAPI：生成 API + WebSocket 进度 + 人在环决策
    models/schemas.py    # Pydantic 数据契约（请求 / DiffReport / 状态机）
    core/
      preprocess.py      # 文字约束抽取（中文启发式，可换 LLM）
      comparison.py      # 六维一致性比对引擎（产品核心）
      orchestrator.py    # 图文联合生成 + 自检 + 修正闭环编排
    adapters/
      base.py            # Hunyuan3D 适配器接口
      mock.py            # Mock：无 GPU 产出真实网格，全链路可跑可测
      http.py            # 真实混元 3D 服务 HTTP 适配器骨架
    store.py             # 进程内任务存储 + 进度广播
  tests/                 # 几何度量 / 收敛 / API 端到端测试
frontend/
  index.html app.js styles.css   # 多图上传 + 文字输入 + 进度 + 报告 + 3D 预览
  vendor/                        # 本地 three.js（离线可运行）
```

## 快速开始

### 方式一：Docker（推荐，一条命令）
```bash
docker compose up --build
# 打开 http://127.0.0.1:8000
```

### 方式二：本地（Make / 脚本）
```bash
make install && make dev     # 或直接 ./scripts/dev.sh
# 打开 http://127.0.0.1:8000
```

上传至少 2 张图片（正图必填），填写文字描述，选模态引导权重，点「立即生成」。
可观察进度 `图文联合生成 → 一致性自检 → 定向修正 → 完成`，报告从「待修正」收敛到
「已一致」，并在右侧预览生成的 3D 模型。

配置（复制 `.env.example` 为 `.env` 按需改）：`TO3D_ADAPTER`（mock/http）、
`TO3D_HUNYUAN_ENDPOINT`、`TO3D_OUTPUT_DIR`。

## 测试

```bash
make test        # 或 cd backend && python -m pytest -q   → 11 passed
```

覆盖：中文约束抽取、网格底部/比例的真实几何测量、自检偏差判定与冲突仲裁、
图文联合生成与修正闭环收敛、REST + 产物下载端到端。CI 见 `.github/workflows/ci.yml`
（测试 + Docker 构建冒烟）。

## 接入真实混元 3D 服务

实现并部署 Hunyuan3D 推理服务（条件编码 + DiT + Paint，图文联合条件生成），
暴露约定端点，然后：

```bash
export TO3D_ADAPTER=http
export TO3D_HUNYUAN_ENDPOINT=https://your-hunyuan3d-service
```

接口与 Mock 完全一致，编排 / 自检 / 前端均无需改动。

- **可直接运行的桥接服务**（把官方 Hunyuan3D-2 包装成本项目接口）：
  [`serving/`](serving/README.md) —— 含安装混元、下载权重、启动、连接主项目的完整步骤。
- **图文联合条件的训练/adapter 方案**：
  [`docs/接入混元3D与联合条件方案.md`](docs/接入混元3D与联合条件方案.md)。

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/generation` | 创建生成任务 |
| GET  | `/api/v1/generation/{id}` | 查询任务状态与差异报告 |
| POST | `/api/v1/generation/{id}/decision` | 人在环决策 |
| GET  | `/api/v1/generation/{id}/mesh` | 下载 GLB |
| WS   | `/ws/tasks/{id}` | 实时进度推送 |

## 当前实现范围（M1 MVP）

已打通「图 + 文联合生成 → 六维自检 → 修正闭环 → 输出 + 预览」完整链路，
几何维度为真实测量。后续（M2+）：接入真实混元 3D 的双模态 CFG 联合条件生成、
CLIP/VQA 花纹检测、PBR 材质分析、局部几何重生成、按品类自动配权，详见方案文档里程碑。
