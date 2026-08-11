from pathlib import Path

import pytest

import pi_model
from pi_model import PI0
from pi_model import checkpoint_asset_id


def write_norm_stats(assets_dir: Path, asset_id: str) -> None:
    path = assets_dir / asset_id / "norm_stats.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")


def test_checkpoint_asset_id_preserves_nested_dataset_id(tmp_path: Path):
    assets_dir = tmp_path / "assets"
    write_norm_stats(assets_dir, "Shiki42/parallelvla_putcab_clean_verified_v2_50")

    assert checkpoint_asset_id(assets_dir) == "Shiki42/parallelvla_putcab_clean_verified_v2_50"


@pytest.mark.parametrize("asset_ids", [(), ("first", "second")])
def test_checkpoint_asset_id_requires_exactly_one_norm_stats(
    tmp_path: Path,
    asset_ids: tuple[str, ...],
):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    for asset_id in asset_ids:
        write_norm_stats(assets_dir, asset_id)

    with pytest.raises(ValueError, match="exactly one"):
        checkpoint_asset_id(assets_dir)


def test_pi0_loads_the_explicit_checkpoint_directory(tmp_path: Path, monkeypatch):
    checkpoint_dir = tmp_path / "external" / "checkpoint" / "30000"
    write_norm_stats(checkpoint_dir / "assets", "dataset/id")
    captured = {}
    monkeypatch.setattr(pi_model._config, "get_config", lambda name: name)  # noqa: SLF001

    def create_policy(config, path, *, robotwin_repo_id):
        captured.update(config=config, path=path, asset_id=robotwin_repo_id)
        return object()

    monkeypatch.setattr(pi_model._policy_config, "create_trained_policy", create_policy)  # noqa: SLF001

    PI0("train_config", checkpoint_dir, 50)

    assert captured == {
        "config": "train_config",
        "path": checkpoint_dir.resolve(),
        "asset_id": "dataset/id",
    }
