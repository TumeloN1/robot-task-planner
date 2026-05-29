"""LLM client wrapper (Google Gemini API; default model is Gemma 4 31B).

Uses JSON-mime mode (response_mime_type="application/json") rather than
response_json_schema: Gemma open models do not support schema mode (it returns
504 DEADLINE_EXCEEDED), and JSON-mime + prompt-described shape + robust parsing
works across both Gemma and Gemini. Robust parse: json.loads the text (stripping
markdown fences if present), validate into a Plan, with one repair retry.
"""

from __future__ import annotations

import json
import os
import time

from rtp.perception.api import SceneState
from rtp.planner.prompt import build_planning_prompt
from rtp.planner.schema import Plan

DEFAULT_MODEL = "gemma-4-31b-it"
_TRANSIENT_STATUS = (503, 504, 429)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rsplit("```", 1)[0]
    return t.strip()


class GeminiPlanner:
    def __init__(self, *, model: str | None = None, temperature: float = 0.2,
                 max_parse_repairs: int = 1, request_timeout_ms: int = 30000,
                 max_server_retries: int = 2) -> None:
        self.model = model or os.environ.get("RTP_GEMINI_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.max_parse_repairs = max_parse_repairs
        self.max_server_retries = max_server_retries
        from google import genai
        from google.genai import types

        # Disable the SDK's own retries so they don't compound with ours; we
        # handle transient 5xx/429 with bounded backoff in _generate().
        self._client = genai.Client(http_options=types.HttpOptions(
            timeout=request_timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        ))

    def _generate(self, prompt: str) -> str:
        from google.genai import errors

        last: Exception | None = None
        for attempt in range(self.max_server_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": self.temperature,
                    },
                )
                return resp.text or ""
            except errors.ServerError as e:
                last = e
                if getattr(e, "code", None) not in _TRANSIENT_STATUS \
                        or attempt >= self.max_server_retries:
                    raise
                time.sleep(2.0 * (attempt + 1))  # linear backoff on transient errors
        raise last if last else RuntimeError("unreachable")

    def _parse(self, text: str) -> Plan:
        return Plan.model_validate(json.loads(_strip_fences(text)))

    def propose(self, instruction: str, scene: SceneState,
                failure_context: str | None = None) -> Plan:
        """Generate a validated `Plan` for the instruction and scene."""
        prompt = build_planning_prompt(instruction, scene, failure_context)
        text = self._generate(prompt)
        for attempt in range(self.max_parse_repairs + 1):
            try:
                return self._parse(text)
            except Exception as e:
                if attempt >= self.max_parse_repairs:
                    raise
                text = self._generate(
                    prompt + f"\n\nYour previous output failed to parse ({e}). "
                    "Return ONLY a single valid JSON object matching the format."
                )
        raise RuntimeError("unreachable")
