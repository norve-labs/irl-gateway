"""`irl-gateway init`: signup -> register -> saved credentials + MCP config."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from irl_gateway import init as init_module
from irl_gateway.init import InitError, run_init


class FakeServer:
    def __init__(self, signup=(201, None), register=(201, None)):
        self.calls: list[tuple[str, dict, str | None]] = []
        self._signup = signup
        self._register = register

    def __call__(self, url, body, token):
        self.calls.append((url, body, token))
        if url.endswith("/irl/signup"):
            status, payload = self._signup
            return status, payload or {"token": "tok-123", "token_id": "abc", "tier": "paper"}
        status, payload = self._register
        return status, payload or {"agent_id": "agent-1", "status": "Active"}


def test_init_signs_up_registers_and_prints_config(tmp_path):
    server = FakeServer()
    lines: list[str] = []

    rc = run_init(
        [
            "--server",
            "https://irl.test/",
            "--name",
            "bot",
            "--contact",
            "me@x.io",
            "--home",
            str(tmp_path),
        ],
        post=server,
        out=lines.append,
    )

    assert rc == 0
    (signup_url, signup_body, signup_token), (reg_url, reg_body, reg_token) = server.calls
    assert signup_url == "https://irl.test/irl/signup" and signup_token is None
    assert signup_body == {"client_name": "bot", "contact": "me@x.io"}
    assert reg_url == "https://irl.test/irl/agents" and reg_token == "tok-123"
    assert reg_body["allowed_venues"] == ["paper-binance"]
    assert reg_body["allowed_assets"] == ["BTC/USDT", "ETH/USDT"]
    assert len(reg_body["model_hash_hex"]) == 64

    saved = json.loads((tmp_path / "agent.json").read_text())
    assert saved["env"]["IRL_API_TOKEN"] == "tok-123"
    assert saved["env"]["IRL_AGENT_ID"] == "agent-1"
    assert saved["env"]["IRL_MODEL_HASH"] == reg_body["model_hash_hex"]
    text = "\n".join(lines)
    assert "claude mcp add irl-gateway" in text
    assert saved["env"]["IRL_BASE_URL"] == "https://irl.test"
    assert "tok-123" not in text  # the server reads the token from agent.json


def test_init_explains_a_server_without_signup(tmp_path):
    server = FakeServer(signup=(404, {"error": "SIGNUP_DISABLED"}))
    with pytest.raises(InitError, match="doesn't offer self-serve signup"):
        run_init(["--home", str(tmp_path)], post=server, out=lambda _: None)
    assert len(server.calls) == 1  # never tries to register


def test_init_surfaces_the_quota_message(tmp_path):
    server = FakeServer(signup=(429, {"error": "QUOTA_EXCEEDED", "message": "try again tomorrow"}))
    with pytest.raises(InitError, match="try again tomorrow"):
        run_init(["--home", str(tmp_path)], post=server, out=lambda _: None)


def test_init_rejects_malformed_assets(tmp_path):
    with pytest.raises(InitError, match="--assets"):
        run_init(
            ["--assets", "BTCUSDT", "--home", str(tmp_path)], post=FakeServer(), out=lambda _: None
        )


# ---- robust output, idempotence, uvx path --------------------------------------

AGENT_UUID = "ff6cb58a-90ff-424e-a67f-fa6f626fe44b"


def _uuid_server() -> FakeServer:
    return FakeServer(register=(201, {"agent_id": AGENT_UUID, "status": "Active"}))


def _cp1252_stdout(monkeypatch) -> io.TextIOWrapper:
    # What Windows gives a redirected/piped stdout when PYTHONIOENCODING is unset.
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    return stream


def _captured(stream: io.TextIOWrapper) -> str:
    stream.flush()
    return stream.buffer.getvalue().decode("cp1252")


def test_init_output_survives_a_cp1252_stdout(tmp_path, monkeypatch):
    # Reproduces the funnel crash: UnicodeEncodeError on '\u2713' after signup.
    stream = _cp1252_stdout(monkeypatch)

    rc = run_init(["--home", str(tmp_path)], post=_uuid_server(), out=print)

    assert rc == 0
    assert "mcpServers" in _captured(stream)


def test_init_messages_are_ascii(tmp_path, monkeypatch):
    # The temp dir (and the user's name in it) may not be ASCII: judge only
    # the messages, with the home path taken out.
    monkeypatch.setenv("PATH", "")
    home = tmp_path / "home"
    lines: list[str] = []

    run_init(["--home", str(home)], post=_uuid_server(), out=lines.append)

    assert "\n".join(lines).replace(str(home), "<home>").isascii()


def test_main_survives_non_ascii_input_on_a_cp1252_stdout(tmp_path, monkeypatch):
    stream = _cp1252_stdout(monkeypatch)
    home = tmp_path / "\u65e5\u672c"  # a home path cp1252 can't encode

    rc = init_module.main(["--home", str(home), "--name", "bot-\u2713"], post=_uuid_server())

    assert rc == 0
    assert "mcpServers" in _captured(stream)
    assert (home / "agent.json").exists()


def test_main_reports_init_errors_on_stderr(tmp_path, capsys):
    server = FakeServer(signup=(429, {"message": "try again tomorrow"}))

    rc = init_module.main(["--home", str(tmp_path)], post=server)

    captured = capsys.readouterr()
    assert rc == 1
    assert "try again tomorrow" in captured.err and captured.out == ""


def test_saved_credentials_come_first_and_the_mcp_config_has_no_token(tmp_path):
    lines: list[str] = []

    run_init(["--home", str(tmp_path)], post=_uuid_server(), out=lines.append)

    assert lines[0].startswith("Saved to ") and str(tmp_path / "agent.json") in lines[0]
    assert "tok-123" not in "\n".join(lines)  # the token stays in agent.json


def test_init_is_idempotent_when_agent_json_is_valid(tmp_path):
    run_init(["--home", str(tmp_path)], post=_uuid_server(), out=lambda _: None)
    before = (tmp_path / "agent.json").read_text()
    again = _uuid_server()
    lines: list[str] = []

    rc = run_init(["--home", str(tmp_path)], post=again, out=lines.append)

    assert rc == 0
    assert again.calls == []  # no second signup
    assert (tmp_path / "agent.json").read_text() == before
    text = "\n".join(lines)
    assert AGENT_UUID in text and "--force" in text and "mcpServers" in text
    assert "tok-123" not in text


def _backups(home) -> list:
    return sorted(home.glob("agent.json.*.bak"))


def test_force_creates_a_new_agent_and_keeps_every_previous_token(tmp_path):
    run_init(["--home", str(tmp_path)], post=_uuid_server(), out=lambda _: None)
    first = (tmp_path / "agent.json").read_text()
    again = _uuid_server()

    rc = run_init(["--home", str(tmp_path), "--force"], post=again, out=lambda _: None)
    second = (tmp_path / "agent.json").read_text()
    run_init(["--home", str(tmp_path), "--force"], post=_uuid_server(), out=lambda _: None)

    assert rc == 0
    assert len(again.calls) == 2
    assert second != first
    backups = _backups(tmp_path)
    assert len(backups) == 2  # two --force runs don't overwrite each other
    assert {b.read_text() for b in backups} == {first, second}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_failed_save_leaves_the_previous_agent_json_intact(tmp_path, monkeypatch):
    run_init(["--home", str(tmp_path)], post=_uuid_server(), out=lambda _: None)
    before = (tmp_path / "agent.json").read_text()
    real_replace = os.replace

    def failing_replace(src, dst):
        if str(src).endswith(".tmp"):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(init_module.os, "replace", failing_replace)

    with pytest.raises(InitError, match="could not save"):
        run_init(["--home", str(tmp_path), "--force"], post=_uuid_server(), out=lambda _: None)

    assert (tmp_path / "agent.json").read_text() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_init_uses_irl_gateway_home_when_no_home_is_given(tmp_path, monkeypatch):
    home = tmp_path / "gw-home"
    monkeypatch.setenv("IRL_GATEWAY_HOME", str(home))
    lines: list[str] = []

    run_init([], post=_uuid_server(), out=lines.append)

    assert (home / "agent.json").exists()
    assert _mcp_block(lines)["mcpServers"]["irl-gateway"]["env"] == {
        "IRL_GATEWAY_HOME": str(home)
    }


def test_existence_check_permission_error_is_an_init_error(tmp_path, monkeypatch):
    saved = tmp_path / "agent.json"
    real_stat = Path.stat

    def stat_(self, *args, **kwargs):
        if self == saved:
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_)
    server = _uuid_server()

    with pytest.raises(InitError, match="could not read"):
        run_init(["--home", str(tmp_path)], post=server, out=lambda _: None)
    assert server.calls == []


def test_an_invalid_agent_json_is_not_overwritten_without_force(tmp_path):
    (tmp_path / "agent.json").write_text('{"env": {"IRL_API_TOKEN": "keep-me"', encoding="utf-8")
    server = _uuid_server()

    with pytest.raises(InitError, match="--force") as info:
        run_init(["--home", str(tmp_path)], post=server, out=lambda _: None)

    assert server.calls == []
    assert "keep-me" not in str(info.value)
    assert "keep-me" in (tmp_path / "agent.json").read_text()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_agent_json_is_private_from_creation(tmp_path):
    run_init(["--home", str(tmp_path)], post=_uuid_server(), out=lambda _: None)
    assert stat.S_IMODE((tmp_path / "agent.json").stat().st_mode) == 0o600


def _mcp_block(lines: list[str]) -> dict:
    text = "\n".join(lines)
    return json.loads(text[text.index("{") : text.rindex("}") + 1])


def _fake_uvx(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / ("uvx.exe" if sys.platform == "win32" else "uvx")
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


def test_mcp_config_uses_the_absolute_uvx_path(tmp_path, monkeypatch):
    real = _fake_uvx(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(real.parent))
    monkeypatch.setattr(init_module, "_default_home", lambda: tmp_path / "home")
    lines: list[str] = []

    run_init(["--home", str(tmp_path / "home")], post=_uuid_server(), out=lines.append)

    entry = _mcp_block(lines)["mcpServers"]["irl-gateway"]
    assert entry == {"command": str(real), "args": ["irl-gateway"]}


def test_a_uvx_planted_in_the_cwd_is_never_printed(tmp_path, monkeypatch):
    cwd = tmp_path / "untrusted"
    _fake_uvx(cwd)
    real = _fake_uvx(tmp_path / "bin")
    monkeypatch.chdir(cwd)
    # cwd first via every spelling a PATH can carry it, then the real one.
    entries = ["", ".", "untrusted", str(cwd), str(real.parent)]
    monkeypatch.setenv("PATH", os.pathsep.join(entries))
    lines: list[str] = []

    run_init(["--home", str(tmp_path / "home")], post=_uuid_server(), out=lines.append)

    assert _mcp_block(lines)["mcpServers"]["irl-gateway"]["command"] == str(real)


def test_only_an_untrusted_uvx_falls_back_to_bare_uvx_with_a_warning(tmp_path, monkeypatch):
    cwd = tmp_path / "untrusted"
    _fake_uvx(cwd)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", str(cwd)]))
    lines: list[str] = []

    run_init(["--home", str(tmp_path / "home")], post=_uuid_server(), out=lines.append)

    entry = _mcp_block(lines)["mcpServers"]["irl-gateway"]
    assert entry["command"] == "uvx"
    text = "\n".join(lines)
    assert "Warning" in text and "absolute path" in text
    # A non-default home has to reach the server, or it won't find agent.json.
    assert entry["env"] == {"IRL_GATEWAY_HOME": str(tmp_path / "home")}
    assert "IRL_GATEWAY_HOME=" in text
