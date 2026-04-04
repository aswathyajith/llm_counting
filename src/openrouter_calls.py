"""OpenRouter chat completions (OpenAI-compatible API).

Inputs (function `chat_completion`):
  api_key   — OpenRouter API key. Prefer env var OPENROUTER_API_KEY; pass explicitly
              to override.
  model     — Model id, e.g. "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet".
  messages  — Chat turns: [{"role": "user"|"system"|"assistant", "content": "..."}, ...].

Optional:
  temperature, max_tokens — forwarded to the API when set.
  site_url, app_name      — set HTTP-Referer / X-Title for OpenRouter rankings (optional).
  extra                   — any extra top-level JSON fields for the request body.

Config file (JSON or TOML) — keys recognized by the CLI (all optional except you must end
up with model + messages from config and/or CLI):
  api_key, model, temperature, max_tokens, site_url, app_name
  extra                   — object merged into the request body (before CLI --extra-json).
  prompt, system          — optional default user / system strings for the CLI entrypoint.

Environment:
  OPENROUTER_API_KEY — used when `api_key` is omitted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Keys loaded from config / merged with CLI (values passed through to chat_completion).
_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "model",
        "temperature",
        "max_tokens",
        "site_url",
        "app_name",
        "extra",
        "prompt",
        "system",
    }
)


def load_openrouter_config(path: str | Path) -> dict[str, Any]:
    """Load OpenRouter settings from a `.json` or `.toml` file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".toml", ".tml"):
        try:
            tomllib = importlib.import_module("tomllib")
        except ModuleNotFoundError as e:
            raise RuntimeError("TOML config requires Python 3.11+") from e
        data = tomllib.loads(text)
    else:
        raise ValueError(f"Unsupported config type (use .json or .toml): {p}")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object / TOML table")
    return data


def chat_completion(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    site_url: str | None = None,
    app_name: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """POST /chat/completions; returns parsed JSON (raises on HTTP errors)."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or pass api_key=")

    body: dict[str, Any] = {"model": model, "messages": list(messages)}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if extra:
        body.update(dict(extra))

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_CHAT_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    if site_url:
        req.add_header("HTTP-Referer", site_url)
    if app_name:
        req.add_header("X-Title", app_name)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {err_body}") from e


def assistant_text(response: Mapping[str, Any]) -> str:
    """First message content string from a chat completion response."""
    choice = response["choices"][0]
    msg = choice["message"]
    return msg["content"]


def _merge_config_and_cli(
    file_cfg: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Config first; CLI flags that were passed override. Merges `extra` dicts."""
    merged: dict[str, Any] = dict(file_cfg or {})
    for k, v in vars(args).items():
        if k in ("config", "extra_json") or k.startswith("_"):
            continue
        merged[k] = v
    ex_cfg = merged.get("extra")
    if ex_cfg is not None and not isinstance(ex_cfg, dict):
        raise TypeError("config 'extra' must be an object/table")
    ex_cli = getattr(args, "extra_json", None)
    if ex_cli is not None:
        if not isinstance(ex_cli, dict):
            raise TypeError("--extra-json must decode to a JSON object")
        merged["extra"] = {**(ex_cfg or {}), **ex_cli}
    elif ex_cfg is not None:
        merged["extra"] = dict(ex_cfg)
    return merged


if __name__ == "__main__":
    s = argparse.SUPPRESS
    p = argparse.ArgumentParser(
        description="OpenRouter chat call. Load defaults from --config; CLI overrides file.",
    )
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        metavar="PATH",
        help="JSON or TOML file (api_key, model, temperature, ...)",
    )
    p.add_argument("--model", default=s, help="Model id, e.g. openai/gpt-4o-mini")
    p.add_argument("--prompt", default=s, help="User message (or set prompt in config)")
    p.add_argument("--system", default=s, help="Optional system message")
    p.add_argument(
        "--api-key", default=s, metavar="KEY", dest="api_key", help="Override API key"
    )
    p.add_argument("--temperature", type=float, default=s)
    p.add_argument("--max-tokens", type=int, default=s, dest="max_tokens")
    p.add_argument("--site-url", default=s, dest="site_url", metavar="URL")
    p.add_argument("--app-name", default=s, dest="app_name", metavar="NAME")
    p.add_argument(
        "--extra-json",
        default=s,
        dest="extra_json",
        metavar='{"k":"v"}',
        type=json.loads,
        help="JSON object merged into request body (overrides config extra keys)",
    )
    args = p.parse_args()

    file_cfg: dict[str, Any] | None = None
    if args.config is not None:
        file_cfg = load_openrouter_config(args.config)

    merged = _merge_config_and_cli(file_cfg, args)

    unknown = set(merged) - _CONFIG_KEYS
    if unknown:
        raise SystemExit(f"Unknown config key(s): {sorted(unknown)}")

    prompt = merged.get("prompt")
    if prompt is None:
        p.error("Provide --prompt or prompt in config")

    model = merged.get("model")
    if not model:
        p.error("Provide --model or model in config")

    msgs: list[dict[str, str]] = []
    system = merged.get("system")
    if system:
        msgs.append({"role": "system", "content": str(system)})
    msgs.append({"role": "user", "content": str(prompt)})

    extra = merged.get("extra")
    if extra is not None and not isinstance(extra, dict):
        raise TypeError("merged 'extra' must be a dict")

    out = chat_completion(
        model=str(model),
        messages=msgs,
        api_key=(str(merged["api_key"]) if merged.get("api_key") else None),
        temperature=merged.get("temperature"),
        max_tokens=merged.get("max_tokens"),
        site_url=merged.get("site_url"),
        app_name=merged.get("app_name"),
        extra=extra,
    )
    print(assistant_text(out))
