#!/usr/bin/env python3
"""Adversarial-review client for Z.AI GLM — the campaign's SECOND reviewer.

Dual adversarial review is standing policy in this repo (main ``a6060d41d``):
every substantive change gets codex AND a second model.  This is the second
model's transport.

WHY THIS EXISTS RATHER THAN THE ``mcp__zai__ask_glm`` TOOL
----------------------------------------------------------
Measured 2026-08-14, and worth writing down because the failure looked like an
expired key for a whole session:

* ``Z_AI_API_KEY`` (in ``~/.bashrc``, and the one inherited by the MCP server)
  returns **401 Authentication Failed** on every endpoint and every auth scheme
  tried — bearer on ``api.z.ai``, bearer on ``open.bigmodel.cn``, and a signed
  JWT.  It is simply not a live key.
* ``ZAI_API_KEY`` (in ``~/.bash_profile``, note the missing underscore) DOES
  authenticate, but on the general endpoint ``/api/paas/v4/`` it returns
  ``{"code": "1113", "message": "Insufficient balance or no resource package"}``
  for every model.
* The same key on the **coding** endpoint ``/api/coding/paas/v4/`` returns
  **200**.  The account carries a coding plan, not a general API package.

So the working combination is ``ZAI_API_KEY`` + the coding endpoint, which is
what this client uses.  It reads the key from the environment first and falls
back to parsing ``~/.bash_profile`` so a fresh shell is not required.

GLM ON THIS ENDPOINT IS A REASONING MODEL and currently answers as ``glm-5.3``.
It spends most of its budget in ``reasoning_content``; a 1500-token cap produced
1497 reasoning tokens and an EMPTY ``content``.  ``--max-tokens`` therefore
defaults high, and the client reports the split so a truncated review is
obvious rather than silently empty.

A NOTE ON WHAT TO ASK IT
------------------------
Its reasoning trace shows it does not reliably recall FV3 internals (it guessed
at routine names unprompted).  Give it the SOURCE and the contract; do not ask
it to remember the model.  That is the same discipline the campaign applies to
itself.
"""
from __future__ import annotations

import argparse
import http.client as http_client
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

CODING_ENDPOINT = "https://api.z.ai/api/coding/paas/v4/chat/completions"
GENERAL_ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"
_PROFILE = pathlib.Path.home() / ".bash_profile"


def resolve_key() -> str:
    """``ZAI_API_KEY`` from the environment, else parsed from ~/.bash_profile.

    Deliberately does NOT fall back to ``Z_AI_API_KEY``: that variable is set on
    this machine and is dead (401), so silently using it would reproduce the
    exact confusion this client documents.
    """
    key = os.environ.get("ZAI_API_KEY", "").strip()
    if key:
        return key
    if _PROFILE.is_file():
        text = _PROFILE.read_text(errors="ignore")
        m = re.search(r"ZAI_API_KEY[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9._-]{20,})",
                      text)
        if m:
            return m.group(1)
    raise SystemExit(
        "no ZAI_API_KEY in the environment or ~/.bash_profile. NOTE: "
        "Z_AI_API_KEY (with the underscore) is a DIFFERENT, dead key on this "
        "machine and is not used as a fallback on purpose.")


def ask(prompt: str, *, model: str = "glm-5.2", max_tokens: int = 32000,
        endpoint: str = CODING_ENDPOINT, timeout: float = 1800.0,
        stream: bool = True) -> dict:
    """One completion.  Returns the parsed message plus usage.

    STREAMS BY DEFAULT, and that is not a preference.  A non-streaming request
    for a long answer from this reasoning model died with
    ``http.client.IncompleteRead(3279 bytes read)`` — the connection is dropped
    while the server is still thinking, because nothing crosses the wire for
    minutes.  Streaming keeps the socket busy and, just as usefully, means a
    partial answer survives a mid-flight drop instead of being lost entirely.

    Raises SystemExit with the server's own message on an API error — a
    ``1113`` balance error and a ``401`` auth error mean very different things
    and must not be collapsed into "GLM is down".
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(
        endpoint, data=body,
        headers={"Authorization": f"Bearer {resolve_key()}",
                 "Content-Type": "application/json"})
    try:
        fh = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:  # surface the server's own text
        raise SystemExit(
            f"GLM HTTP {exc.code}: {exc.read().decode(errors='ignore')[:400]}"
        ) from exc

    if not stream:
        payload = json.load(fh)
        if "error" in payload:
            raise SystemExit(f"GLM API error: {payload['error']}")
        msg = payload["choices"][0]["message"]
        return {"served_model": payload.get("model"),
                "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "usage": payload.get("usage", {}), "truncated": False}

    content, reasoning, usage, served = [], [], {}, None
    truncated = False
    try:
        with fh:
            for raw in fh:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    raise SystemExit(f"GLM API error: {chunk['error']}")
                served = served or chunk.get("model")
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices", ()):
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
    except (http_client.IncompleteRead, ConnectionError, TimeoutError,
            urllib.error.URLError) as exc:
        # Keep what arrived.  A partial review is still reviewable; silently
        # returning nothing is what the non-streaming path used to do.
        truncated = True
        print(f"# WARNING: stream ended early ({type(exc).__name__}); "
              f"keeping {sum(map(len, content))} chars of answer",
              file=sys.stderr)

    return {"served_model": served, "content": "".join(content),
            "reasoning": "".join(reasoning), "usage": usage,
            "truncated": truncated}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-file", required=True,
                    help="file holding the review prompt (use a file, not an "
                         "argv string: shell quoting mangles code excerpts)")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--general-endpoint", action="store_true",
                    help="use /api/paas/v4/ instead of the coding endpoint "
                         "(expected to fail with 1113 on this account)")
    ap.add_argument("--out", help="write the answer here as well as to stdout")
    ap.add_argument("--show-reasoning", action="store_true")
    ap.add_argument("--no-stream", action="store_true",
                    help="disable streaming (expect IncompleteRead on long "
                         "answers -- see ask() docstring)")
    args = ap.parse_args()

    prompt = pathlib.Path(args.prompt_file).read_text()
    res = ask(prompt, model=args.model, max_tokens=args.max_tokens,
              stream=not args.no_stream,
              endpoint=(GENERAL_ENDPOINT if args.general_endpoint
                        else CODING_ENDPOINT))

    u = res["usage"]
    detail = u.get("completion_tokens_details", {}) or {}
    reasoning_tok = detail.get("reasoning_tokens", 0)
    completion_tok = u.get("completion_tokens", 0)
    print(f"# served model: {res['served_model']}")
    print(f"# tokens: prompt {u.get('prompt_tokens')}, completion "
          f"{completion_tok} (of which reasoning {reasoning_tok})")
    if not res["content"]:
        # The exact failure mode this client was written to make visible.
        print("# WARNING: EMPTY content. The budget went entirely to reasoning "
              "-- raise --max-tokens. This is NOT a refusal and NOT an error.",
              file=sys.stderr)
    if args.show_reasoning and res["reasoning"]:
        print("\n===== reasoning =====\n" + res["reasoning"])
    print("\n===== answer =====\n" + res["content"])

    if args.out:
        pathlib.Path(args.out).write_text(res["content"])
    return 0 if res["content"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
