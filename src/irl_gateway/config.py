"""Gateway settings from environment variables, validated at startup.

Required: IRL_BASE_URL, IRL_API_TOKEN, IRL_AGENT_ID, IRL_MODEL_HASH.
When the environment carries none of the agent identity, they come from the
agent.json that `irl-gateway init` writes (see `resolve_env`).
The broker defaults to paper trading at live public prices; a real account
needs GATEWAY_BROKER=exchange plus exchange credentials.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$")


logger = logging.getLogger(__name__)

AGENT_FILE = "agent.json"
# The agent identity is one unit: a token, its agent and the model hash it was
# registered with. It comes whole from the environment or whole from agent.json.
IDENTITY_KEYS = ("IRL_API_TOKEN", "IRL_AGENT_ID", "IRL_MODEL_HASH")
# What agent.json may set. Exchange credentials and the broker choice are
# deliberately absent: a file on disk never switches the gateway to live trading.
FILE_KEYS = frozenset(
    {
        "IRL_BASE_URL",
        *IDENTITY_KEYS,
        "AGENT_MODEL_ID",
        "AGENT_CONFIG_CHECKSUM",
        "IRL_L2_MODE",
        "EXCHANGE_ID",
        "PAPER_BALANCES",
    }
)


class ConfigError(ValueError):
    pass


class NotConfiguredError(ConfigError):
    """Neither the environment nor agent.json holds an agent identity."""


@dataclass(frozen=True)
class Settings:
    irl_base_url: str
    irl_api_token: str
    agent_id: str
    model_hash_hex: str
    default_model_id: str
    hyperparameter_checksum: str
    use_regime_ref: bool
    broker: str  # "paper" | "exchange"
    exchange_id: str
    exchange_api_key: str
    exchange_api_secret: str
    exchange_testnet: bool
    paper_balances: dict[str, float]
    journal_path: Path
    kill_switch_path: Path
    paper_state_path: Path


def load_settings(env: Mapping[str, str]) -> Settings:
    def required(name: str) -> str:
        value = env.get(name, "").strip()
        if not value:
            raise ConfigError(f"{name} is required")
        return value

    agent_id = required("IRL_AGENT_ID")
    if not _UUID.match(agent_id):
        raise ConfigError("IRL_AGENT_ID must be a UUID")
    model_hash = required("IRL_MODEL_HASH")
    if not _HEX64.match(model_hash):
        raise ConfigError(
            "IRL_MODEL_HASH must be 64 hex characters (the hash the agent was registered with)"
        )

    l2 = env.get("IRL_L2_MODE", "off").strip().lower()
    if l2 not in ("off", "regime"):
        raise ConfigError("IRL_L2_MODE must be 'off' or 'regime'")

    broker = env.get("GATEWAY_BROKER", "paper").strip().lower()
    if broker not in ("paper", "exchange"):
        raise ConfigError("GATEWAY_BROKER must be 'paper' or 'exchange'")
    exchange_id = env.get("EXCHANGE_ID", "binance").strip().lower()
    api_key = env.get("EXCHANGE_API_KEY", "").strip()
    api_secret = env.get("EXCHANGE_API_SECRET", "").strip()
    if broker == "exchange" and not (api_key and api_secret):
        raise ConfigError("GATEWAY_BROKER=exchange needs EXCHANGE_API_KEY and EXCHANGE_API_SECRET")

    home = gateway_home(env)
    return Settings(
        irl_base_url=required("IRL_BASE_URL"),
        irl_api_token=required("IRL_API_TOKEN"),
        agent_id=agent_id,
        model_hash_hex=model_hash.lower(),
        default_model_id=env.get("AGENT_MODEL_ID", "unspecified-model").strip()
        or "unspecified-model",
        hyperparameter_checksum=env.get("AGENT_CONFIG_CHECKSUM", "none").strip() or "none",
        use_regime_ref=l2 == "regime",
        broker=broker,
        exchange_id=exchange_id,
        exchange_api_key=api_key,
        exchange_api_secret=api_secret,
        exchange_testnet=_bool(env.get("EXCHANGE_TESTNET", "true")),
        paper_balances=parse_balances(env.get("PAPER_BALANCES", "USDT=1000")),
        journal_path=home / "journal.jsonl",
        kill_switch_path=home / "KILL",
        paper_state_path=home / "paper_state.json",
    )


def parse_balances(text: str) -> dict[str, float]:
    balances: dict[str, float] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        asset, sep, amount = item.partition("=")
        try:
            value = float(amount)
        except ValueError:
            value = -1.0
        if not sep or not asset.strip() or value < 0:
            raise ConfigError(f"PAPER_BALANCES entries look like USDT=1000, got {item!r}")
        balances[asset.strip().upper()] = value
    return balances


def _bool(text: str) -> bool:
    value = text.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"expected a boolean, got {text!r}")


def gateway_home(env: Mapping[str, str]) -> Path:
    configured = env.get("IRL_GATEWAY_HOME", "").strip()
    return Path(configured) if configured else Path.home() / ".irl-gateway"


def resolve_env(
    env: Mapping[str, str], *, warn: Callable[[str], None] = logger.warning
) -> dict[str, str]:
    """The environment, completed from agent.json when it lacks the identity.

    Environment values win over the file; empty values count as unset. The
    file's token is only ever used with the server it was issued for.
    """
    present = [k for k in IDENTITY_KEYS if env.get(k, "").strip()]
    if len(present) == len(IDENTITY_KEYS):
        return dict(env)
    path = gateway_home(env) / AGENT_FILE
    if present:
        missing = [k for k in IDENTITY_KEYS if k not in present]
        raise ConfigError(
            f"{', '.join(present)} set but {', '.join(missing)} missing: set all of "
            f"{', '.join(IDENTITY_KEYS)} in the environment, or none of them to use {path}"
        )
    if not agent_file_exists(path):
        raise NotConfiguredError(f"no agent identity in the environment and no {path}")
    file_env = read_agent_file(path, warn=warn)
    env_url = env.get("IRL_BASE_URL", "").strip().rstrip("/")
    file_url = file_env.get("IRL_BASE_URL", "").strip().rstrip("/")
    if env_url and env_url != file_url:
        raise ConfigError(
            f"IRL_BASE_URL is set to a different server than the one {path} was issued for; "
            "unset it, or set the token, agent and model hash for that server too"
        )
    merged = dict(file_env)
    merged.update({k: v for k, v in env.items() if v.strip()})
    return merged


def agent_file_exists(path: Path) -> bool:
    try:
        path.stat()
    except FileNotFoundError:
        return False
    except OSError as exc:  # e.g. PermissionError on the directory
        raise ConfigError(f"could not read {path}: {_reason(exc)}") from None
    return True


def read_agent_file(path: Path, *, warn: Callable[[str], None] = logger.warning) -> dict[str, str]:
    """The validated `env` block of agent.json. Error messages name keys and
    the path, never values: the file holds a bearer token."""
    _check_permissions(path, warn)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {_reason(exc)}") from None
    try:
        payload = json.loads(text)
    except ValueError:
        raise ConfigError(f"{path} is not valid JSON") from None
    block = payload.get("env") if isinstance(payload, dict) else None
    if not isinstance(block, dict):
        raise ConfigError(f"{path} must hold an object with an 'env' object")
    env: dict[str, str] = {}
    for key, value in block.items():
        if key not in FILE_KEYS:
            raise ConfigError(
                f"{path} sets {key}, which agent.json can't set; use an environment variable"
            )
        if not isinstance(value, str):
            raise ConfigError(f"{path}: {key} must be a string")
        env[key] = value
    missing = [k for k in ("IRL_BASE_URL", *IDENTITY_KEYS) if not env.get(k, "").strip()]
    if missing:
        raise ConfigError(f"{path} is missing {', '.join(missing)}")
    return env


def _check_permissions(path: Path, warn: Callable[[str], None]) -> None:
    """POSIX: refuse a file another user could have written (it chooses where
    the token goes and which agent trades); warn if others can read it."""
    owner_mode = _posix_owner_mode(path)
    if owner_mode is None:
        return  # Windows: access is governed by the profile's ACLs
    mode, uid, me = owner_mode
    problem = _unsafe_reason(mode, uid, me)
    if problem:
        raise ConfigError(
            f"{path} is {problem}; refusing to use it. Check its contents, then make it "
            f"yours and private: chmod 600 {path}"
        )
    if _readable_by_others(mode):
        warn(f"{path} holds your IRL token and is readable by other users; run: chmod 600 {path}")


def _posix_owner_mode(path: Path) -> tuple[int, int, int] | None:
    """(permission bits, file owner uid, current uid), or None on Windows."""
    if sys.platform == "win32":
        return None
    try:
        st = path.stat()
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {_reason(exc)}") from None
    return stat.S_IMODE(st.st_mode), st.st_uid, os.getuid()


def _unsafe_reason(mode: int, uid: int, me: int) -> str | None:
    if uid != me:
        return "owned by another user"
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        return "writable by other users"
    return None


def _readable_by_others(mode: int) -> bool:
    return bool(mode & (stat.S_IRGRP | stat.S_IROTH))


def _reason(exc: OSError) -> str:
    return exc.strerror or type(exc).__name__
