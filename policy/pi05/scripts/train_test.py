import pytest

from scripts import train


def test_update_limit_defaults_to_formal_horizon(monkeypatch):
    monkeypatch.delenv("PARALLELVLA_PREFLIGHT_MAX_UPDATES", raising=False)

    assert train.resolve_update_limit(500, 0) == 500


def test_update_limit_preserves_formal_config_for_save_reload(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_PREFLIGHT_MAX_UPDATES", "2")

    assert train.resolve_update_limit(500, 1) == 2


@pytest.mark.parametrize("value", ["not-an-int", "1", "501"])
def test_update_limit_rejects_invalid_or_nonadvancing_values(monkeypatch, value):
    monkeypatch.setenv("PARALLELVLA_PREFLIGHT_MAX_UPDATES", value)

    with pytest.raises(ValueError, match="preflight update limit"):
        train.resolve_update_limit(500, 1)
