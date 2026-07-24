from openpi.policies import policy_config
from openpi.training import config


def test_with_asset_id_replaces_frozen_config_without_mutation():
    original = config.DataConfig(asset_id="original")

    updated = policy_config._with_asset_id(  # noqa: SLF001
        original,
        "Shiki42/parallelvla_putcab_clean_verified_v2_50",
    )

    assert original.asset_id == "original"
    assert updated.asset_id == "Shiki42/parallelvla_putcab_clean_verified_v2_50"
