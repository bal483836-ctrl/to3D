# to3D · 图文双通道 3D 生成

基于混元 3D（Hunyuan3D）的 **多图 + 文字融合** 3D 生成系统：一份 3D 模型同时由
多视图图片与文字描述共同驱动。核心机制是**双通道生成 → 六维一致性比对 → 定向修正闭环**。

- 方案设计（完整）：[`docs/方案设计.md`](docs/方案设计.md)
- 可视化方案页：[`docs/方案设计.html`](docs/方案设计.html)

## 它是怎么工作的

```
多视图(2~8) + 文字  →  通道A 图生3D(主几何) ┐
                        通道B 文生3D(语义参照) ┴→ 六维比对 → 达标? ─是→ 输出
                                                          └否→ 定向修正(以图为准) ↺
```

1. **图生 3D（通道 A）**：多视图 → 网格 + PBR 纹理，作为交付主体。
2. **文生 3D（通道 B）**：文字 → 三维化，作为「文字理想形态」的语义标尺，不直接交付。
3. **一致性比对**：在 **器型 / 花纹 / 底部 / 高度 / 宽度 / 材质** 六个维度逐一打分。
   - 几何维度直接在网格上真实测量（Chamfer/IoU、包围盒比例、底部形态、对称性）。
   - 纹理/材质维度基于纹理描述与文字约束评分（生产环境由 CLIP/VQA/PBR 分析得到）。
4. **冲突仲裁（以图为准）**：某维度有对应视角图 → 以图为准；无图但文字明确 → 以文校正。
5. **修正闭环**：几何偏差局部重生成、纹理偏差按文字重绘，有界迭代收敛（防发散，取历史最优）。

## 目录结构

```
backend/
  app/
    main.py              # FastAPI：生成 API + WebSocket 进度 + 人在环决策
    models/schemas.py    # Pydantic 数据契约（请求 / DiffReport / 状态机）
    core/
      preprocess.py      # 文字约束抽取（中文启发式，可换 LLM）
      comparison.py      # 六维一致性比对引擎（产品核心）
      orchestrator.py    # 双通道并行 + 修正闭环编排
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

## 本地运行

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt

# 启动（默认使用 Mock 适配器，无需 GPU）
cd backend && uvicorn app.main:app --reload --port 8000
# 打开 http://127.0.0.1:8000
```

上传至少 2 张图片（正图必填），填写文字描述，点「立即生成」。可观察比对报告在
修正后从「待修正」收敛到「已一致」，并在右侧预览生成的 3D 模型。

## 测试

```bash
cd backend && python -m pytest -q      # 10 passed
```

覆盖：中文约束抽取、网格底部/比例的真实几何测量、比对偏差判定与冲突仲裁、
修正闭环收敛、REST + 产物下载端到端。

## 接入真实混元 3D 服务

实现并部署 Hunyuan3D 推理服务（DiT + MV + Paint），暴露约定端点，然后：

```bash
export TO3D_ADAPTER=http
export TO3D_HUNYUAN_ENDPOINT=https://your-hunyuan3d-service
```

约定端点见 [`backend/app/adapters/http.py`](backend/app/adapters/http.py)。
接口与 Mock 完全一致，编排 / 比对 / 前端均无需改动。

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/generation` | 创建生成任务 |
| GET  | `/api/v1/generation/{id}` | 查询任务状态与差异报告 |
| POST | `/api/v1/generation/{id}/decision` | 人在环决策 |
| GET  | `/api/v1/generation/{id}/mesh` | 下载 GLB |
| WS   | `/ws/tasks/{id}` | 实时进度推送 |

## 当前实现范围（M1 MVP）

已打通「图 + 文输入 → 双通道 → 六维比对 → 修正闭环 → 输出 + 预览」完整链路，
几何维度为真实测量。后续（M2+）：接入真实混元 3D、CLIP/VQA 花纹检测、PBR 材质
分析、局部几何重生成、按品类自动配权，详见方案文档里程碑。
