"""`.env` 自动加载。

这是一个真实踩过的坑：`.env.example` 一直写着"复制为 .env"，但没有任何地方读它，
于是填了 `TO3D_ADAPTER=tencent` 也静默跑在 mock 上（日志里 adapter=mock，
产出的是占位花瓶）。这里锁住加载行为与优先级。
"""
from __future__ import annotations

import os

import pytest

from app.dotenv import load_dotenv, parse_env


def test_parse_basic_pairs():
    got = parse_env("TO3D_ADAPTER=tencent\nTENCENT_REGION=ap-guangzhou\n")
    assert got == {"TO3D_ADAPTER": "tencent", "TENCENT_REGION": "ap-guangzhou"}


def test_parse_strips_inline_comment():
    """.env.example 里就是这种写法，不剥掉注释会把它读进值里。"""
    got = parse_env("TENCENT_RESULT_FORMAT=GLB          # GLB|OBJ|STL|USDZ|FBX\n")
    assert got["TENCENT_RESULT_FORMAT"] == "GLB"


def test_parse_keeps_hash_inside_quotes_and_without_space():
    got = parse_env('A="secret#1 with spaces"\nB=pass#notcomment\n')
    assert got["A"] == "secret#1 with spaces"
    assert got["B"] == "pass#notcomment"


def test_parse_handles_export_prefix_and_junk_lines():
    got = parse_env(
        "# 注释行\n\nexport TO3D_MAX_CONCURRENCY=3\n没有等号的垃圾行\n=空键\nEMPTY=\n"
    )
    assert got["TO3D_MAX_CONCURRENCY"] == "3"
    assert got["EMPTY"] == ""
    assert "没有等号的垃圾行" not in got


def test_load_does_not_override_real_env(tmp_path, monkeypatch):
    """12-factor：进程环境里已有的值必须压过 .env（Docker/CI 注入的不能被文件改掉）。"""
    env = tmp_path / ".env"
    env.write_text("TO3D_ADAPTER=tencent\nSOME_NEW_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("TO3D_ADAPTER", "http")
    monkeypatch.delenv("SOME_NEW_KEY", raising=False)

    path, applied = load_dotenv(str(env))
    assert path == str(env) and applied == 1
    assert os.environ["TO3D_ADAPTER"] == "http"      # 未被覆盖
    assert os.environ["SOME_NEW_KEY"] == "from_file"  # 未设置的才写入


def test_load_override_flag(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TO3D_ADAPTER=tencent\n", encoding="utf-8")
    monkeypatch.setenv("TO3D_ADAPTER", "http")
    load_dotenv(str(env), override=True)
    assert os.environ["TO3D_ADAPTER"] == "tencent"


def test_load_missing_file_is_noop(tmp_path):
    path, applied = load_dotenv(str(tmp_path / "nope.env"))
    assert (path, applied) == (None, 0)


def test_env_file_var_selects_path(tmp_path, monkeypatch):
    env = tmp_path / "custom.env"
    env.write_text("PICKED_BY_TO3D_ENV_FILE=1\n", encoding="utf-8")
    monkeypatch.setenv("TO3D_ENV_FILE", str(env))
    monkeypatch.delenv("PICKED_BY_TO3D_ENV_FILE", raising=False)
    path, _ = load_dotenv()
    assert path == str(env)
    assert os.environ["PICKED_BY_TO3D_ENV_FILE"] == "1"


def test_default_path_is_repo_root_not_cwd():
    """make dev 从 backend/ 启动，按相对路径找 .env 会找不到——必须锚在仓库根。"""
    from app.dotenv import DEFAULT_ENV_PATH

    assert os.path.isabs(DEFAULT_ENV_PATH)
    assert DEFAULT_ENV_PATH.endswith(os.path.join(os.sep, ".env"))
    root = os.path.dirname(DEFAULT_ENV_PATH)
    # 仓库根的标志物：.env.example 与 backend/ 都在这里
    assert os.path.isfile(os.path.join(root, ".env.example"))
    assert os.path.isdir(os.path.join(root, "backend"))


def test_settings_read_env_after_dotenv_load(tmp_path, monkeypatch):
    """配置在导入时求值，加载顺序反了就等于没加载——重新导入一次锁住顺序。"""
    import importlib

    env = tmp_path / ".env"
    env.write_text("TO3D_ADAPTER=tencent\nTO3D_MAX_CONCURRENCY=7\n", encoding="utf-8")
    monkeypatch.setenv("TO3D_ENV_FILE", str(env))
    for key in ("TO3D_ADAPTER", "TO3D_MAX_CONCURRENCY"):
        monkeypatch.delenv(key, raising=False)

    import app.config as config_mod

    reloaded = importlib.reload(config_mod)
    try:
        assert reloaded.ENV_FILE == str(env)
        assert reloaded.settings.adapter == "tencent"
        assert reloaded.settings.max_concurrency == 7
    finally:
        # 还原全局配置，避免污染其它测试
        monkeypatch.undo()
        importlib.reload(config_mod)


def test_warns_on_non_tencent_credential_format(monkeypatch):
    """填了别家服务的 ak-/sk- 密钥时，启动即指出，别等跑完一轮才 AuthFailure。"""
    from app import config

    monkeypatch.setattr(config.settings, "adapter", "tencent")
    monkeypatch.setattr(config.settings, "tencent_secret_id", "ak-20260811-a67b8a96")
    monkeypatch.setattr(config.settings, "tencent_cos_bucket", "bucket-1250000000")
    warns = config.config_warnings()
    assert len(warns) == 1 and "AKID" in warns[0]


def test_no_warning_for_valid_tencent_credential(monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "adapter", "tencent")
    monkeypatch.setattr(config.settings, "tencent_secret_id", "AKID" + "z" * 32)
    monkeypatch.setattr(config.settings, "tencent_cos_bucket", "bucket-1250000000")
    assert config.config_warnings() == []


def test_warns_when_cos_bucket_missing(monkeypatch):
    """多图需公网 URL，没配桶会静默退化成单图——这种降级必须说出来。"""
    from app import config

    monkeypatch.setattr(config.settings, "adapter", "tencent")
    monkeypatch.setattr(config.settings, "tencent_secret_id", "AKID" + "z" * 32)
    monkeypatch.setattr(config.settings, "tencent_cos_bucket", "")
    warns = config.config_warnings()
    assert len(warns) == 1 and "COS" in warns[0]


def test_warnings_only_apply_to_tencent(monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "adapter", "mock")
    monkeypatch.setattr(config.settings, "tencent_secret_id", "ak-whatever")
    monkeypatch.setattr(config.settings, "tencent_cos_bucket", "")
    assert config.config_warnings() == []


def test_startup_without_env_file(monkeypatch, tmp_path):
    """没有 .env 时（全新 clone 的默认情形）也必须能正常启动并提示如何配置。"""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr("app.config.ENV_FILE", None)
    monkeypatch.delenv("TO3D_ENV_FILE", raising=False)
    with TestClient(app) as client:  # lifespan 会走"未找到 .env"分支
        assert client.get("/api/health").status_code == 200


def test_config_exposes_default_env_path():
    """main.py 的提示信息从 app.config 取该常量，别只留在 app.dotenv 里。"""
    from app import config

    assert config.DEFAULT_ENV_PATH


@pytest.mark.parametrize("adapter", ["mock", "tencent", "http"])
def test_health_reports_config_source(adapter, monkeypatch):
    """能一眼看出「.env 有没有被读到、是不是还在 mock」，不必翻日志。"""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr("app.main.settings.adapter", adapter)
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert body["adapter"] == adapter
    assert body["is_mock"] is (adapter == "mock")
    assert "env_file" in body
