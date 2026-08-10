# CLAUDE.md — 项目上下文

## 这是什么
基于混元 3D（Hunyuan3D）的**图文联合 3D 生成**系统：多视图图片与文字描述**同时（联合）**
作为条件，**一次生成一份**模型；再用六维一致性对该产物自检并定向修正。
关键：是"图文联合生成一份模型"，**不是**"图生一个 + 文生一个再比较"。

## 目录
- `backend/` FastAPI 服务：编排 + 六维自检 + 适配器（mock / 真实 http）
  - `app/core/orchestrator.py` 联合生成→自检→修正闭环
  - `app/core/comparison.py` 六维一致性自检（几何为真实网格测量）
  - `app/adapters/{base,mock,http}.py` 适配器接口 / 无 GPU Mock / 真实服务 HTTP
  - `app/config.py` 环境变量配置
- `frontend/` 单页应用：多图上传 + 文字 + 模态引导权重 + 进度 + 报告 + three.js 预览
  （`frontend/vendor/` 为本地 three.js，离线可运行）
- `tools/` Blender 50 视角渲染脚本 + 相机参数转换(camera_utils，可单测)
- `datasets/` 参考 `reference_meta_data.json`(50 组相机参数，800×800)
- `docs/` 方案设计、可视化页、接入混元/腾讯云、50 视角数据集生成
- `backend/Dockerfile` `docker-compose.yml` `Makefile` `scripts/dev.sh` 运行/部署

## 常用命令
```bash
make install      # 建 venv + 装依赖
make dev          # 本地热重载 http://127.0.0.1:8000
make test         # 运行测试（应 22 passed）
make docker-up    # 容器启动 http://127.0.0.1:8000
./scripts/dev.sh  # 一条命令起本地服务
```

## 关键约定
- 适配器三选一：`TO3D_ADAPTER=mock`(无GPU) | `tencent`(腾讯云API,无需GPU,
  见 `docs/接入腾讯云API.md`) | `http`(自建GPU桥接,见 `docs/接入混元3D与联合条件方案.md`)。
- 六维：器型/花纹/底部/高度/宽度/材质。冲突仲裁：有对应视角图→以图为准；
  无图但文字明确→提高文字引导修正。
- 修改后务必 `make test` 保持通过；改前端交互建议用 Playwright 走一遍真实流程。

## 边界
- 当前几何维度为真实测量；花纹/材质为可插拔描述评分，接真实模型时替换适配器
  与这两个维度的评分器即可，其余复用。
