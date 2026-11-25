"""The MCP surface: tool list, safety annotations and structured results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from irl_gateway import __version__
from irl_gateway import server as server_module
from irl_gateway.brokers import PaperBroker
from irl_gateway.config import Settings, load_settings
from irl_gateway.server import build_gateway, create_server

from .fakes import AGENT_ID, MODEL_HASH, make_gateway


async def _call(server, name: str, args: dict | None = None) -> dict:
    result = await server.call_tool(name, args or {})
    assert not result.is_error, result.content
    return result.structured_content


async def test_only_execute_trade_is_destructive(tmp_path):
    gateway, _, _ = make_gateway(tmp_path)

    tools = {t.name: t for t in await create_server(gateway).list_tools()}

    assert set(tools) == {
        "execute_trade",
        "get_quote",
        "get_balances",
        "get_policy",
        "get_trace",
        "list_recent_trades",
    }
    assert tools["execute_trade"].annotations.destructive_hint is True
    assert tools["execute_trade"].input_schema["required"] == ["symbol", "side", "rationale"]
    for name, tool in tools.items():
        if name != "execute_trade":
            assert tool.annotations.read_only_hint is True, name


async def test_execute_trade_returns_the_structured_outcome(tmp_path):
    gateway, irl, _ = make_gateway(tmp_path)
    server = create_server(gateway)

    outcome = await _call(
        server,
        "execute_trade",
        {"symbol": "BTC/USDT", "side": "buy", "rationale": "test entry", "notional": 50.0},
    )

    assert outcome["status"] == "filled"
    assert outcome["verdict"] == "MATCHED"
    assert outcome["fill"]["quantity"] == 0.001
    assert len(irl.authorize_calls) == 1


async def test_read_tools(tmp_path):
    gateway, _, _ = make_gateway(tmp_path)
    server = create_server(gateway)
    await _call(
        server,
        "execute_trade",
        {"symbol": "BTC/USDT", "side": "buy", "rationale": "r", "quantity": 0.001},
    )
    (tmp_path / "KILL").write_text("")

    quote = await _call(server, "get_quote", {"symbol": "BTC/USDT"})
    balances = await _call(server, "get_balances")
    policy = await _call(server, "get_policy")
    trace = await _call(server, "get_trace", {"trace_id": "trace-1"})
    trades = await _call(server, "list_recent_trades", {"limit": 500})

    assert quote == {"symbol": "BTC/USDT", "price": 50_000.0}
    assert balances["venue_id"] == "paper-binance" and balances["balances"]["BTC"] == 0.001
    assert policy["max_notional"] == 200.0 and policy["allowed_assets"] == ["BTC/USDT"]
    assert policy["kill_switch_engaged"] is True
    assert "model_hash_hex" not in policy
    assert trace["verification_status"] == "MATCHED"
    assert trades["trades"][0]["status"] == "filled"
    assert trades["trades"][0]["context"]["rationale"] == "r"


async def test_build_gateway_defaults_to_paper_on_public_prices(tmp_path):
    settings = load_settings(
        {
            "IRL_BASE_URL": "http://irl:4000",
            "IRL_API_TOKEN": "tok",
            "IRL_AGENT_ID": AGENT_ID,
            "IRL_MODEL_HASH": MODEL_HASH,
            "IRL_GATEWAY_HOME": str(tmp_path),
            "PAPER_BALANCES": "USDT=250",
        }
    )

    gateway = build_gateway(settings)
    try:
        assert isinstance(gateway.broker, PaperBroker)
        assert gateway.broker.venue_id == "paper-binance"
        assert await gateway.broker.get_balances() == {"USDT": 250.0}
        assert gateway.journal.path == tmp_path / "journal.jsonl"
    finally:
        await gateway.broker.close()
        await gateway.irl.close()


# ---- process entry point: unconfigured, --help, --version, agent.json ----------

_IRL_VARS = ("IRL_BASE_URL", "IRL_API_TOKEN", "IRL_AGENT_ID", "IRL_MODEL_HASH")


_CONFIG_PREFIXES = ("IRL_", "GATEWAY_", "EXCHANGE_", "AGENT_", "PAPER_")


def _clean_env(home: Path) -> dict[str, str]:
    # No gateway configuration leaks in from the developer's or CI's shell.
    env = {k: v for k, v in os.environ.items() if not k.startswith(_CONFIG_PREFIXES)}
    env.update(
        IRL_GATEWAY_HOME=str(home),
        HOME=str(home),
        USERPROFILE=str(home),
        PYTHONIOENCODING="",
    )
    return env


def _run_cli(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "irl_gateway", *args],
        check=False,
        env=_clean_env(home),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_unconfigured_server_says_what_to_run_without_a_traceback(tmp_path):
    result = _run_cli(tmp_path)

    assert result.returncode == 2
    assert result.stdout == ""  # stdout belongs to the MCP protocol
    assert "Traceback" not in result.stderr
    assert "uvx irl-gateway init" in result.stderr
    assert len(result.stderr.strip().splitlines()) <= 2


def test_help_and_version_work_without_config(tmp_path):
    helped = _run_cli(tmp_path, "--help")
    version = _run_cli(tmp_path, "--version")

    assert helped.returncode == 0 and "init" in helped.stdout and "Traceback" not in helped.stderr
    assert version.returncode == 0 and __version__ in version.stdout


def test_invalid_config_is_one_clear_message(tmp_path, monkeypatch, capsys):
    (tmp_path / "agent.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "agent.json").chmod(0o600)
    for name in _IRL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IRL_GATEWAY_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["irl-gateway"])

    with pytest.raises(SystemExit) as info:
        server_module.main()

    captured = capsys.readouterr()
    assert info.value.code == 2
    assert captured.out == ""
    assert "not valid JSON" in captured.err and "uvx irl-gateway init" in captured.err


def test_server_starts_from_agent_json_alone(tmp_path, monkeypatch, capsys):
    (tmp_path / "agent.json").write_text(
        json.dumps(
            {
                "env": {
                    "IRL_BASE_URL": "https://norve.dev",
                    "IRL_API_TOKEN": "tok-file",
                    "IRL_AGENT_ID": AGENT_ID,
                    "IRL_MODEL_HASH": MODEL_HASH,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "agent.json").chmod(0o600)  # POSIX refuses a group-writable file
    for name in _IRL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IRL_GATEWAY_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["irl-gateway"])
    seen: list[Settings] = []
    ran: list[str] = []

    class FakeServer:
        def run(self, transport: str) -> None:
            ran.append(transport)

    monkeypatch.setattr(server_module, "build_gateway", lambda s: seen.append(s) or "gw")
    monkeypatch.setattr(server_module, "create_server", lambda gw: FakeServer())

    server_module.main()

    assert seen[0].irl_api_token == "tok-file" and seen[0].agent_id == AGENT_ID
    assert ran == ["stdio"]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("tool", "args", "method", "expected"),
    [
        (
            "get_quote",
            {"symbol": "BTC/USDT"},
            "get_price",
            "could not price BTC/USDT (ExchangeError)",
        ),
        ("get_balances", {}, "get_balances", "could not read balances (ExchangeError)"),
    ],
)
async def test_read_tools_do_not_pass_venue_text_to_the_agent(
    tmp_path, tool, args, method, expected
):
    import ccxt.async_support as ccxt_async

    from .test_gateway import _VENUE_BODY, _BrokerRaisingOn

    _, _, paper = make_gateway(tmp_path)
    gateway, _, _ = make_gateway(
        tmp_path, broker=_BrokerRaisingOn(paper, method, ccxt_async.ExchangeError(_VENUE_BODY))
    )

    result = await _call(create_server(gateway), tool, args)

    assert result == {"error": expected}


async def test_irl_read_tools_return_only_code_and_status(tmp_path):
    from irl_gateway.irl import IrlUnavailable

    from .fakes import FakeIrl

    class _FailingIrl(FakeIrl):
        async def get_agent(self, agent_id):
            raise IrlUnavailable(502, "BAD_GATEWAY", "<html>upstream body</html>")

        async def get_trace(self, trace_id):
            raise IrlUnavailable(502, "BAD_GATEWAY", "<html>upstream body</html>")

    gateway, _, _ = make_gateway(tmp_path, _FailingIrl())
    server = create_server(gateway)

    assert await _call(server, "get_policy") == {"error": "[BAD_GATEWAY] (HTTP 502)"}
    assert await _call(server, "get_trace", {"trace_id": "t"}) == {
        "error": "[BAD_GATEWAY] (HTTP 502)"
    }
