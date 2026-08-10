"""请求 schema：最多 8 张图、最少 2 张、必含正图。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.schemas import GenerationRequest, ImageInput, ViewName

PX = "data:image/png;base64,QUJD"


def _imgs(views):
    return [ImageInput(view=v, url=PX, required=(v == ViewName.front)) for v in views]


def test_eight_views_enum():
    assert len(list(ViewName)) == 8


def test_accepts_up_to_8():
    req = GenerationRequest(images=_imgs(list(ViewName)), prompt="花瓶")
    assert len(req.images) == 8


def test_rejects_more_than_8():
    with pytest.raises(ValidationError):
        GenerationRequest(images=_imgs(list(ViewName)) + [_imgs([ViewName.top])[0]], prompt="x")


def test_requires_min_2_and_front():
    with pytest.raises(ValidationError):
        GenerationRequest(images=_imgs([ViewName.front]), prompt="x")  # 少于 2
    with pytest.raises(ValidationError):
        GenerationRequest(images=_imgs([ViewName.left, ViewName.right]), prompt="x")  # 缺正图
