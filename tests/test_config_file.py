"""Fallback from the environment to the agent.json written by `irl-gateway init`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from irl_gateway import config
from irl_gateway.config import ConfigError, NotConfiguredError, load_settings, resolve_env

from .fakes import AGENT_ID, MODEL_HASH

TOKEN = "irl_secret_token_value_do_not_leak"
FILE_ENV = {
    "IRL_BASE_URL": "https://norve.dev",
    "IRL_API_TOKEN": TOKEN,
    "IRL_AGENT_ID": AGENT_ID,
    "IRL_MODEL_HASH": MODEL_HASH,
    "AGENT_MODEL_ID": "claude-opus-5-5",
    "EXCHANGE_ID": "binance",
    "PAPER_BALANCES": "USDT=1000",
}


def _write(home: Path, payload: object) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "agent.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    return path


def test_without_env_or_file_it_is_not_configured(tmp_path):
    with pytest.raises(NotConfiguredError):
        resolve_env({"IRL_GATEWAY_HOME": str(tmp_path)})


def test_agent_json_fills_in_when_the_environment_is_empty(tmp_path):
    _write(tmp_path, {"tier": "paper", "env": FILE_ENV})

    settings = load_settings(resolve_env({"IRL_GATEWAY_HOME": str(tmp_path)}))

    assert settings.irl_api_token == TOKEN
    assert settings.agent_id == AGENT_ID
    assert settings.irl_base_url == "https://norve.dev"
    assert settings.default_model_id == "claude-opus-5-5"
    assert settings.journal_path == tmp_path / "journal.jsonl"


def test_empty_env_values_count_as_unset(tmp_path):
    # Registry-driven installers often pass the declared variables as "".
    _write(tmp_path, {"env": FILE_ENV})
    env = {"IRL_GATEWAY_HOME": str(tmp_path), "IRL_API_TOKEN": "", "IRL_AGENT_ID": "  "}

    assert load_settings(resolve_env(env)).irl_api_token == TOKEN


def test_environment_takes_priority_over_the_file(tmp_path):
    _write(tmp_path, {"env": FILE_ENV})
    other_agent = "11111111-2222-3333-4444-555555555555"
    env = {
        "IRL_GATEWAY_HOME": str(tmp_path),
        "IRL_BASE_URL": "http://irl:4000",
        "IRL_API_TOKEN": "env-token",
        "IRL_AGENT_ID": other_agent,
        "IRL_MODEL_HASH": "b" * 64,
        "PAPER_BALANCES": "USDT=50",
    }

    settings = load_settings(resolve_env(env))

    assert settings.irl_api_token == "env-token"
    assert settings.agent_id == other_agent
    assert settings.irl_base_url == "http://irl:4000"
    assert settings.paper_balances == {"USDT": 50.0}
    # A full identity in the environment never needs the file.
    assert settings.default_model_id == "unspecified-model"


def test_non_identity_env_overrides_the_file(tmp_path):
    _write(tmp_path, {"env": FILE_ENV})
    env = {"IRL_GATEWAY_HOME": str(tmp_path), "PAPER_BALANCES": "USDT=50"}

    settings = load_settings(resolve_env(env))

    assert settings.irl_api_token == TOKEN
    assert settings.paper_balances == {"USDT": 50.0}


def test_a_partial_identity_in_the_environment_is_not_mixed_with_the_file(tmp_path):
    _write(tmp_path, {"env": FILE_ENV})
    env = {"IRL_GATEWAY_HOME": str(tmp_path), "IRL_AGENT_ID": AGENT_ID}

    with pytest.raises(ConfigError, match="IRL_API_TOKEN") as info:
        resolve_env(env)
    assert TOKEN not in str(info.value)


def test_the_file_token_is_never_sent_to_a_different_server(tmp_path):
    _write(tmp_path, {"env": FILE_ENV})
    env = {"IRL_GATEWAY_HOME": str(tmp_path), "IRL_BASE_URL": "https://elsewhere.example"}

    with pytest.raises(ConfigError, match="IRL_BASE_URL"):
        resolve_env(env)


def test_same_server_with_trailing_slash_is_fine(tmp_path):
    _write(tmp_path, {"env": FILE_ENV})
    env = {"IRL_GATEWAY_HOME": str(tmp_path), "IRL_BASE_URL": "https://norve.dev/"}

    assert resolve_env(env)["IRL_API_TOKEN"] == TOKEN


@pytest.mark.parametrize(
    "payload, message",
    [
        (f'{{"env": {{"IRL_API_TOKEN": "{TOKEN}"', "not valid JSON"),
        ([1, 2], "env"),
        ({"env": "x"}, "env"),
        ({"env": {"IRL_API_TOKEN": 5}}, "IRL_API_TOKEN"),
        ({"env": {**FILE_ENV, "EXCHANGE_API_SECRET": "s"}}, "EXCHANGE_API_SECRET"),
        ({"env": {**FILE_ENV, "IRL_MODEL_HASH": ""}}, "IRL_MODEL_HASH"),
    ],
)
def test_a_broken_agent_json_is_a_clear_error_without_the_token(tmp_path, payload, message):
    path = _write(tmp_path, payload)

    with pytest.raises(ConfigError, match=message) as info:
        resolve_env({"IRL_GATEWAY_HOME": str(tmp_path)})

    assert not isinstance(info.value, NotConfiguredError)
    assert TOKEN not in str(info.value)
    assert str(path) in str(info.value)


def test_unreadable_agent_json_is_a_config_error(tmp_path):
    (tmp_path / "agent.json").mkdir()  # a directory: exists but can't be read as a file

    with pytest.raises(ConfigError, match="could not read"):
        resolve_env({"IRL_GATEWAY_HOME": str(tmp_path)})


def test_shared_file_warns_without_revealing_the_token(tmp_path, monkeypatch):
    _write(tmp_path, {"env": FILE_ENV})
    monkeypatch.setattr(config, "_readable_by_others", lambda _path: True)
    warnings: list[str] = []

    resolve_env({"IRL_GATEWAY_HOME": str(tmp_path)}, warn=warnings.append)

    assert len(warnings) == 1
    assert "chmod 600" in warnings[0]
    assert TOKEN not in warnings[0]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_posix_permission_check(tmp_path):
    path = _write(tmp_path, {"env": FILE_ENV})
    assert config._readable_by_others(path) is False
    path.chmod(0o644)
    assert config._readable_by_others(path) is True


def test_non_posix_permission_check_is_skipped(tmp_path, monkeypatch):
    path = _write(tmp_path, {"env": FILE_ENV})
    monkeypatch.setattr(config, "_IS_POSIX", False)
    assert config._readable_by_others(path) is False


def test_gateway_home_defaults_to_the_user_home(monkeypatch):
    assert config.gateway_home({}) == Path.home() / ".irl-gateway"
    assert config.gateway_home({"IRL_GATEWAY_HOME": "/x/y"}) == Path("/x/y")
