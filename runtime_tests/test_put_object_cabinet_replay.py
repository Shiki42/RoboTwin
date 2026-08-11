from __future__ import annotations

from pathlib import Path
from runpy import run_path

import pytest


parse_replay_object_identity = run_path(
    str(Path(__file__).parents[1] / "envs/replay_scene.py")
)["parse_replay_object_identity"]


def test_parse_replay_object_identity_uses_recorded_model() -> None:
    scene_info = {
        "info": {
            "{A}": "073_rubikscube/base2",
            "{B}": "036_cabinet/base0",
        }
    }

    assert parse_replay_object_identity(None) is None
    assert parse_replay_object_identity(scene_info) == ("073_rubikscube", 2)


@pytest.mark.parametrize(
    "scene_info",
    (
        {},
        "not-a-dictionary",
        {"info": {}},
        {"info": {"{A}": "073_rubikscube"}},
        {"info": {"{A}": "073_rubikscube/base-x"}},
    ),
)
def test_parse_replay_object_identity_rejects_invalid_metadata(scene_info) -> None:
    with pytest.raises(ValueError):
        parse_replay_object_identity(scene_info)
