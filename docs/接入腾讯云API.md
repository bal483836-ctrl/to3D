# 接入腾讯云「混元生3D」API（无需自建 GPU）

用腾讯云托管的混元生3D(ai3d)出模型：**不用 GPU、按次计费、数据发腾讯**。
上层编排/自检/前端不变，只需开通服务、填密钥、设 `TO3D_ADAPTER=tencent`。

与自建 GPU 桥接(`serving/`)是二选一的两条路，见 README 对比表。

---

## 1. 开通与获取密钥
1. 登录腾讯云控制台，开通 **混元生3D / AI 3D（ai3d）** 服务。
2. 访问管理 → API 密钥，创建 **SecretId / SecretKey**（建议用子账号 + 最小权限）。
3. 确认地域（如 `ap-guangzhou`）与服务可用性、计费方式。

## 2. 安装依赖
```bash
pip install -r backend/requirements-tencent.txt   # 装 tencentcloud-sdk-python-common
```

## 3. 配置
```bash
export TO3D_ADAPTER=tencent
export TENCENT_SECRET_ID=<你的SecretId>
export TENCENT_SECRET_KEY=<你的SecretKey>
export TENCENT_REGION=ap-guangzhou
export TENCENT_RESULT_FORMAT=GLB      # 前端预览用 GLB
# 可选：TENCENT_ENABLE_PBR / TENCENT_POLL_INTERVAL / TENCENT_POLL_TIMEOUT
```
（或写入 `.env`，见 `.env.example` 的 TENCENT_* 段。）

## 4. 启动与验证
```bash
make dev                                   # 或 docker compose up
curl http://127.0.0.1:8000/api/ready       # adapter 应为 tencent
```
前端正常上传图 + 文字生成；产物为腾讯云返回的真实模型，六维自检照常
（几何维度在返回网格上真实测量）。

---

## 5. 工作原理（本适配器做了什么）
`backend/app/adapters/tencent.py`：
1. `SubmitHunyuanTo3DJob` 提交作业（组装 Prompt / 图片 / 多视图 / 格式 / PBR）。
2. 轮询 `QueryHunyuanTo3DJob` 直到 `DONE`。
3. 下载结果模型文件，载入为网格返回给编排层。
鉴权由官方 SDK 的 `CommonClient` 完成（TC3-HMAC-SHA256 签名）。

## 6. 使用限制与注意
- **多视图需公网可访问的图片 URL**：`MultiViewImages` 走 `ViewImageUrl`。
  若前端传的是本地图片(data-uri)，仅支持**单图**(`ImageBase64`)。生产建议先把
  用户图片上传对象存储拿到 URL，再提交多视图（可在 `_build_submit_params` 扩展）。
- **正交视角**：仅 front/back/left/right 映射到腾讯 ViewType，45°/顶/底忽略。
- **无局部编辑**：`refine_geometry` 为 no-op、`repaint` 仅更新纹理描述；几何要贴合
  文字建议补图或用 `text_correct` 策略。
- **字段/版本以文档为准**：Action、Version(`2025-05-13`)、字段名如与你账号下当前
  文档不一致，集中在 `_build_submit_params` / `_parse_result` / 配置项调整即可。
- **计费与配额**：按作业计费，注意并发配额；`TO3D_MAX_CONCURRENCY` 控制我方并发。
- **网络**：容器/服务器需能出网访问 `ai3d.tencentcloudapi.com` 与结果文件下载域名。

## 7. 三条出模型路线对比
| | mock | tencent(云API) | http(自建GPU桥接) |
|---|---|---|---|
| GPU | 否 | 否 | 是 |
| 费用 | 无 | 按次 | 机器/电 |
| 数据 | 本地 | 发腾讯 | 本地 |
| 联合条件 | 完整(模拟) | 云端能力为准 | 务实路线(可训练至完整) |
| 适用 | 开发/演示 | 快速上线/无GPU | 私有化/大规模自控 |
