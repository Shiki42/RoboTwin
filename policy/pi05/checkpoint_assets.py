from pathlib import Path


def checkpoint_asset_id(assets_dir: str | Path) -> str:
    assets_path = Path(assets_dir)
    norm_stats_paths = sorted(assets_path.rglob("norm_stats.json"))
    if len(norm_stats_paths) != 1:
        raise ValueError(
            f"expected exactly one norm_stats.json under {assets_path}, "
            f"found {len(norm_stats_paths)}"
        )
    return norm_stats_paths[0].parent.relative_to(assets_path).as_posix()
