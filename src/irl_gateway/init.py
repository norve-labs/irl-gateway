"""`irl-gateway init`: from nothing to a working MCP config in one command.

1. Get a free paper-tier token from the IRL server (POST /irl/signup).
2. Register an agent with a starter mandate (POST /irl/agents).
3. Save the credentials to ~/.irl-gateway/agent.json and print the MCP
   config to paste into Claude Code, Claude Desktop or any MCP client. The
   server reads agent.json itself, so the printed config carries no token.

Running it again reuses a valid agent.json instead of signing up twice;
`--force` creates a new agent (the old file is kept as agent.json.<timestamp>.bak).

Paper-tier tokens can only trade on paper venues; ask the server's operator
for a full token to trade live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shlex
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from irl_gateway.config import (
    AGENT_FILE,
    ConfigError,
    agent_file_exists,
    gateway_home,
    load_settings,
    read_agent_file,
)

DEFAULT_SERVER = "https://norve.dev"
BACKUP_PATTERN = f"{AGENT_FILE}.<timestamp>.bak"

PostJson = Callable[[str, dict[str, Any], str | None], tuple[int, dict[str, Any]]]


class InitError(Exception):
    pass


def post_json(url: str, body: dict[str, Any], token: str | None) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json", "User-Agent": "irl-gateway-init"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read() or b"{}")
        except ValueError:
            payload = {}
        return exc.code, payload
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise InitError(f"could not reach {url}: {exc}") from exc


def _default_home() -> Path:
    """Where the server looks for agent.json when IRL_GATEWAY_HOME is unset."""
    return Path.home() / ".irl-gateway"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="irl-gateway init",
        description="Get a free paper-trading token, register an agent, print the MCP config.",
    )
    p.add_argument(
        "--server", default=DEFAULT_SERVER, help=f"IRL server (default {DEFAULT_SERVER})"
    )
    p.add_argument("--name", default="my-agent", help="agent name (default my-agent)")
    p.add_argument("--contact", help="optional email or handle, so the operator can reach you")
    p.add_argument(
        "--max-notional",
        type=float,
        default=1000.0,
        help="per-order cap in quote currency (default 1000)",
    )
    p.add_argument("--assets", default="BTC/USDT,ETH/USDT", help="comma-separated allowed symbols")
    p.add_argument(
        "--exchange", default="binance", help="exchange whose prices and rules paper trading uses"
    )
    p.add_argument(
        "--model-id", default="claude-opus-5-5", help="model name sealed into each trace"
    )
    p.add_argument(
        "--home",
        default=str(gateway_home(os.environ)),
        help="where to save agent.json (default $IRL_GATEWAY_HOME or ~/.irl-gateway)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=f"sign up a new agent even if agent.json exists (keeps {BACKUP_PATTERN})",
    )
    return p


def run_init(
    argv: list[str], *, post: PostJson = post_json, out: Callable[[str], None] = print
) -> int:
    args = _parser().parse_args(argv)
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    if not assets or any("/" not in a for a in assets):
        raise InitError("--assets must look like BTC/USDT,ETH/USDT")
    if not args.max_notional > 0:
        raise InitError("--max-notional must be positive")
    # Absolute, so the IRL_GATEWAY_HOME we may print works from any directory.
    home = Path(args.home).expanduser().absolute()
    saved = home / AGENT_FILE

    if _exists(saved) and not args.force:
        env = _existing_env(saved)
        out(f"Found {saved}: agent {env['IRL_AGENT_ID']} on {env['IRL_BASE_URL']}.")
        out("  Not signing up again. To create a new agent instead, re-run with --force.")
        _print_client_config(home, out)
        return 0

    env, tier = _signup_and_register(args, assets, post)
    backup = _save(saved, json.dumps({"tier": tier, "env": env}, indent=2))
    # The path first: if anything below fails, the user still knows where it is.
    out(f"Saved to {saved} (keep it private: it holds your token).")
    if backup is not None:
        out(f"  Previous config kept as {backup}: it holds the old token, delete it")
        out("  or revoke that token once you no longer need it.")
    out(f"OK: paper-tier token issued and agent '{args.name}' registered ({env['IRL_AGENT_ID']}).")
    out(
        f"  Mandate: {', '.join(assets)} on paper-{args.exchange}, "
        f"at most {args.max_notional:g} per order."
    )
    _print_client_config(home, out)
    return 0


def _existing_env(saved: Path) -> dict[str, str]:
    try:
        env = read_agent_file(saved)
        load_settings(env)
    except ConfigError as exc:  # messages name keys and paths, never the token
        raise InitError(
            f"{saved} exists but isn't a valid gateway config ({exc}). Re-run with --force "
            f"to sign up again (the old file is kept as {BACKUP_PATTERN})."
        ) from None
    return env


def _exists(path: Path) -> bool:
    try:
        return agent_file_exists(path)
    except ConfigError as exc:
        raise InitError(str(exc)) from None


def _signup_and_register(
    args: argparse.Namespace, assets: list[str], post: PostJson
) -> tuple[dict[str, str], str | None]:
    server = args.server.rstrip("/")
    signup_body: dict[str, Any] = {"client_name": args.name}
    if args.contact:
        signup_body["contact"] = args.contact
    status, body = post(f"{server}/irl/signup", signup_body, None)
    if status == 404 and body.get("error") == "SIGNUP_DISABLED":
        raise InitError(f"{server} doesn't offer self-serve signup; ask its operator for a token")
    if status != 201:
        raise InitError(f"signup failed (HTTP {status}): {body.get('message') or body}")
    token = body["token"]

    # A fresh identity for this agent. If you later change its model or
    # configuration in a way auditors should see, register a new agent.
    model_hash = hashlib.sha256(
        f"irl-gateway-init:{args.name}:{secrets.token_hex(16)}".encode()
    ).hexdigest()
    status, agent = post(
        f"{server}/irl/agents",
        {
            "name": args.name,
            "model_hash_hex": model_hash,
            "max_notional": args.max_notional,
            "allowed_assets": assets,
            "allowed_venues": [f"paper-{args.exchange}"],
        },
        token,
    )
    if status != 201:
        raise InitError(
            f"agent registration failed (HTTP {status}): {agent.get('message') or agent}"
        )
    env = {
        "IRL_BASE_URL": server,
        "IRL_API_TOKEN": token,
        "IRL_AGENT_ID": agent["agent_id"],
        "IRL_MODEL_HASH": model_hash,
        "AGENT_MODEL_ID": args.model_id,
        "EXCHANGE_ID": args.exchange,
        "PAPER_BALANCES": "USDT=1000",
    }
    return env, body.get("tier")


def _save(path: Path, text: str) -> Path | None:
    """Replace agent.json without ever losing the previous one.

    The new content is fully written (owner-only on POSIX, from the first
    byte) before anything moves; the previous file goes to a timestamped
    backup, so repeated --force runs keep every older token; if the final
    move fails, the previous file is put back. Returns the backup, if any.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp")
    backup: Path | None = None
    try:
        tmp.unlink(missing_ok=True)  # an existing file would keep its old mode
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        if _exists(path):
            backup = _backup_path(path)
            os.replace(path, backup)
        try:
            os.replace(tmp, path)
        except OSError:
            if backup is not None:
                os.replace(backup, path)
            raise
    except OSError as exc:
        raise InitError(
            f"could not save {path} ({exc.strerror or type(exc).__name__}); the new token "
            "was issued but not saved, and any previous agent.json is unchanged"
        ) from None
    finally:
        tmp.unlink(missing_ok=True)
    return backup


def _backup_path(path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = path.with_name(f"{path.name}.{stamp}.bak")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{stamp}-{counter}.bak")
        counter += 1
    return candidate


def _trusted_uvx() -> str | None:
    """Absolute path of uvx from PATH, never from the current directory.

    On Windows `shutil.which("uvx")` looks in the cwd first, even with an
    explicit `path=`; a uvx planted there must not end up in a client config.
    So skip empty, relative and cwd entries, and look each remaining directory
    up with a directory part, which disables the implicit cwd lookup.
    """
    try:
        cwd: str | None = os.path.normcase(os.path.realpath(os.getcwd()))
    except OSError:
        cwd = None
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        entry = raw.strip().strip('"')
        if not entry or not os.path.isabs(entry):
            continue  # "", "." and relative entries resolve against the cwd
        if os.path.normcase(os.path.realpath(entry)) == cwd:
            continue
        found = shutil.which(os.path.join(entry, "uvx"))
        if found and os.path.isabs(found):
            return found
    return None


def _print_client_config(home: Path, out: Callable[[str], None]) -> None:
    # GUI clients (Claude Desktop on macOS in particular) don't inherit the
    # shell's PATH, so a bare "uvx" fails with `spawn uvx ENOENT`: give them the
    # absolute path. Claude Code runs from the user's shell, where "uvx" resolves.
    uvx = _trusted_uvx()
    if uvx is None:
        out("")
        out('Warning: no uvx found on PATH (outside the current directory). Replace "uvx"')
        out("below with its absolute path: `which uvx` (macOS/Linux) or `where uvx` (Windows).")
    entry: dict[str, Any] = {"command": uvx or "uvx", "args": ["irl-gateway"]}
    env_flags: list[str] = []
    if home.resolve() != _default_home().resolve():
        # The server only finds agent.json elsewhere if it's told where.
        entry["env"] = {"IRL_GATEWAY_HOME": str(home)}
        env_flags = ["-e", shlex.quote(f"IRL_GATEWAY_HOME={home}")]
    claude_cmd = " ".join(
        ["claude", "mcp", "add", "irl-gateway", *env_flags, "--", "uvx", "irl-gateway"]
    )
    out("")
    out("Claude Code, one command:")
    out(f"  {claude_cmd}")
    out("")
    out("Any MCP client (Claude Desktop, Cursor, ...), add to its config:")
    out(json.dumps({"mcpServers": {"irl-gateway": entry}}, indent=2))
    out("")
    out("The server reads your token from agent.json; environment variables override it.")
    out("Then ask your agent to call get_policy, and make its first paper trade.")


def _make_stdout_safe() -> None:
    # A redirected stdout on Windows is often cp1252: never let a name or path
    # the console can't encode crash init after the signup already happened.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")


def main(argv: list[str], *, post: PostJson = post_json) -> int:
    _make_stdout_safe()
    try:
        return run_init(argv, post=post)
    except InitError as exc:
        print(f"irl-gateway init: {exc}", file=sys.stderr)
        return 1
