from __future__ import annotations

import re


def parse_replay_object_identity(
    replay_scene_info: dict | None,
) -> tuple[str, int] | None:
    if replay_scene_info is None:
        return None
    if not isinstance(replay_scene_info, dict):
        raise ValueError("replay_scene_info must be a dictionary")
    info = replay_scene_info.get("info")
    if not isinstance(info, dict) or not isinstance(info.get("{A}"), str):
        raise ValueError("replay_scene_info must contain string info['{A}']")
    match = re.fullmatch(r"([^/]+)/base([0-9]+)", info["{A}"])
    if match is None:
        raise ValueError(f"invalid replay object identity: {info['{A}']!r}")
    return match.group(1), int(match.group(2))
