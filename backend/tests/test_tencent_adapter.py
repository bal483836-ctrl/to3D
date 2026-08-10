"""腾讯云 ai3d 适配器测试：参数构建 / 轮询 / 结果解析（不触真实网络）。"""
from __future__ import annotations

import trimesh

from app.core.preprocess import extract_constraints
from app.models.schemas import FusionConfig, ImageInput, ViewName


def _make_adapter(monkeypatch, script):
    """构造适配器并注入伪造的云端调用。script: action -> 返回值序列。"""
    from app import config
    monkeypatch.setattr(config.settings, "tencent_secret_id", "id")
    monkeypatch.setattr(config.settings, "tencent_secret_key", "key")
    from app.adapters import tencent as tmod
    monkeypatch.setattr(tmod.settings, "tencent_secret_id", "id")
    monkeypatch.setattr(tmod.settings, "tencent_secret_key", "key")
    monkeypatch.setattr(tmod.settings, "tencent_poll_interval", 0)

    calls = {"log": []}
    adapter = tmod.TencentCloudAdapter()

    def fake_call(action, params):
        calls["log"].append((action, params))
        seq = script[action]
        return seq.pop(0) if isinstance(seq, list) else seq

    monkeypatch.setattr(adapter, "_call_api", fake_call)
    monkeypatch.setattr(adapter, "_download_mesh", lambda url: trimesh.creation.box())
    return adapter, calls


def test_build_params_multiview_http(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {})
    imgs = [
        ImageInput(view=ViewName.front, url="https://x/f.png", required=True),
        ImageInput(view=ViewName.left, url="https://x/l.png"),
    ]
    p = adapter._build_submit_params(imgs, "青花瓷花瓶")
    assert p["Prompt"] == "青花瓷花瓶"
    assert {v["ViewType"] for v in p["MultiViewImages"]} == {"front", "left"}


def test_build_params_data_uri_single_without_cos(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {})
    from app.adapters import tencent as tmod
    monkeypatch.setattr(tmod.settings, "tencent_cos_bucket", "")  # 未配 COS
    imgs = [
        ImageInput(view=ViewName.front, url="data:image/png;base64,QUJD", required=True),
        ImageInput(view=ViewName.left, url="data:image/png;base64,QUJD"),
    ]
    p = adapter._build_submit_params(imgs, "")
    assert p["ImageBase64"] == "QUJD"        # 退化为单图
    assert "MultiViewImages" not in p


def test_build_params_data_uri_multiview_with_cos(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {})
    from app.adapters import tencent as tmod
    monkeypatch.setattr(tmod.settings, "tencent_cos_bucket", "bkt-123")
    # 伪造上传：返回可辨识的 URL，不触真实 COS
    monkeypatch.setattr(adapter, "_upload_data_uri",
                        lambda view, uri: f"https://cos/{view}.png")
    imgs = [
        ImageInput(view=ViewName.front, url="data:image/png;base64,QUJD", required=True),
        ImageInput(view=ViewName.left, url="data:image/png;base64,QUJD"),
    ]
    p = adapter._build_submit_params(imgs, "青花瓷花瓶")
    assert p["Prompt"] == "青花瓷花瓶"
    assert {v["ViewType"]: v["ViewImageUrl"] for v in p["MultiViewImages"]} == {
        "front": "https://cos/front.png", "left": "https://cos/left.png",
    }


def test_eight_views_with_configured_map(monkeypatch):
    """配置 TENCENT_VIEW_MAP 后，8 个视角(http URL)全部进入 MultiViewImages。"""
    adapter, _ = _make_adapter(monkeypatch, {})
    from app.adapters import tencent as tmod
    monkeypatch.setattr(tmod.settings, "tencent_view_map", {
        "top": "top", "bottom": "bottom", "left45": "left_front", "right45": "right_front",
    })
    views = [ViewName.front, ViewName.back, ViewName.left, ViewName.right,
             ViewName.left45, ViewName.right45, ViewName.top, ViewName.bottom]
    imgs = [ImageInput(view=v, url=f"https://x/{v.value}.png",
                       required=(v == ViewName.front)) for v in views]
    p = adapter._build_submit_params(imgs, "花瓶")
    assert len(p["MultiViewImages"]) == 8
    assert {v["ViewType"] for v in p["MultiViewImages"]} == {
        "front", "back", "left", "right", "top", "bottom", "left_front", "right_front",
    }


def test_default_map_keeps_four_orthogonal(monkeypatch):
    """默认(不配 map)时，只发 4 个正交视角，避免未知 ViewType 报错。"""
    adapter, _ = _make_adapter(monkeypatch, {})
    from app.adapters import tencent as tmod
    monkeypatch.setattr(tmod.settings, "tencent_view_map", {})
    views = [ViewName.front, ViewName.back, ViewName.top, ViewName.bottom]
    imgs = [ImageInput(view=v, url=f"https://x/{v.value}.png",
                       required=(v == ViewName.front)) for v in views]
    p = adapter._build_submit_params(imgs, "")
    assert {v["ViewType"] for v in p["MultiViewImages"]} == {"front", "back"}


def test_parse_result_prefers_format(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {})
    resp = {"ResultFile3Ds": [
        {"Type": "OBJ", "Url": "http://o"}, {"Type": "GLB", "Url": "http://g"},
    ]}
    assert adapter._parse_result(resp) == "http://g"  # 默认 GLB


def test_submit_and_wait_polls_until_done(monkeypatch):
    script = {
        "SubmitHunyuanTo3DJob": {"JobId": "job1"},
        "QueryHunyuanTo3DJob": [
            {"Status": "WAIT"},
            {"Status": "RUN"},
            {"Status": "DONE", "ResultFile3Ds": [{"Type": "GLB", "Url": "http://g"}]},
        ],
    }
    adapter, calls = _make_adapter(monkeypatch, script)
    mesh = adapter._submit_and_wait({"Prompt": "x"})
    assert mesh.is_watertight  # box 是水密的
    # 一次提交 + 三次查询
    actions = [a for a, _ in calls["log"]]
    assert actions == ["SubmitHunyuanTo3DJob", "QueryHunyuanTo3DJob",
                       "QueryHunyuanTo3DJob", "QueryHunyuanTo3DJob"]


def test_generate_returns_mesh_and_texture(monkeypatch):
    script = {
        "SubmitHunyuanTo3DJob": {"JobId": "job1"},
        "QueryHunyuanTo3DJob": {"Status": "DONE",
                                "ResultFile3Ds": [{"Type": "GLB", "Url": "http://g"}]},
    }
    adapter, _ = _make_adapter(monkeypatch, script)
    c = extract_constraints("青花瓷花瓶，缠枝莲纹，釉面高光")
    imgs = [ImageInput(view=ViewName.front, url="https://x/f.png", required=True),
            ImageInput(view=ViewName.right, url="https://x/r.png")]
    res = adapter.generate(imgs, "青花瓷花瓶，缠枝莲纹，釉面高光", c, FusionConfig().guidance())
    assert res.mesh is not None and res.source == "joint"
    assert "缠枝莲" in res.texture.patterns and res.texture.glossy is True
