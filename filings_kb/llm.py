"""LLM backends. Tests use fakes; nothing here runs at import time."""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResult:
    """Text returned by a backend plus the token counts it reported."""

    text: str
    tokens_in: int
    tokens_out: int


class LLMBackend(Protocol):
    """Anything that can turn (system, user, model) into text."""

    def complete(self, system: str, user: str, model: str) -> LLMResult:
        """Run one completion.

        Raises:
            RuntimeError: if the backend call fails.
        """
        ...


class ClaudeCliBackend:
    """Default backend: the ``claude -p`` CLI (uses the author's subscription).

    The user message goes on stdin, because section bodies can exceed the OS
    argv limit. Tools, MCP servers, skills, settings sources and session
    persistence are all disabled so a call carries only our system prompt
    (``--bare`` is unusable: it ignores the subscription's OAuth login).
    Token usage is read from the CLI's JSON output when present, otherwise
    reported as 0; ``tokens_in`` excludes cache tokens.
    """

    def __init__(self, executable: str = "claude", timeout_s: float = 900.0) -> None:
        self._executable = executable
        self._timeout_s = timeout_s

    def complete(self, system: str, user: str, model: str) -> LLMResult:
        """Run ``claude -p`` once.

        Raises:
            RuntimeError: on a non-zero exit, unparseable output, or an
                ``is_error`` result.
        """
        cmd = [
            self._executable, "-p", "--model", model, "--system-prompt", system,
            "--output-format", "json", "--no-session-persistence", "--tools", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--disable-slash-commands", "--setting-sources", "",
        ]
        proc = subprocess.run(
            cmd, input=user, capture_output=True, text=True,
            timeout=self._timeout_s, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr.strip()}")
        try:
            data: dict[str, Any] = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"claude CLI returned non-JSON output: {exc}") from exc
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI reported an error: {data.get('result')}")
        usage = data.get("usage") or {}
        return LLMResult(
            text=str(data.get("result", "")),
            tokens_in=int(usage.get("input_tokens", 0)),
            tokens_out=int(usage.get("output_tokens", 0)),
        )


class AnthropicBackend:
    """Optional backend using the Anthropic API (``pip install '.[anthropic]'``)."""

    def __init__(self, max_tokens: int = 16000) -> None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        import anthropic  # lazy: optional dependency

        self._client = anthropic.Anthropic(api_key=key)
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str, model: str) -> LLMResult:
        """Call the Messages API once and return its text and usage."""
        msg = self._client.messages.create(
            model=model, max_tokens=self._max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResult(text, msg.usage.input_tokens, msg.usage.output_tokens)
