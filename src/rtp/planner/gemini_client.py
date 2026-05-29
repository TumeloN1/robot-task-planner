"""Gemini client wrapper with robust structured-output parsing.

Uses google-genai with response_mime_type="application/json" and
response_json_schema. Robust parse: prefer response.parsed, fall back to
stripping markdown fences then json.loads, then a single repair retry.
"""

from __future__ import annotations

from rtp.planner.schema import Plan


class GeminiPlanner:
    def __init__(self, *, model: str = "gemini-2.5-flash", temperature: float = 0.2,
                 max_parse_repairs: int = 1) -> None:
        self.model = model
        self.temperature = temperature
        self.max_parse_repairs = max_parse_repairs
        # TODO: from google import genai; self.client = genai.Client()
        #       (reads GEMINI_API_KEY from env via python-dotenv).

    def plan(self, prompt: str) -> Plan:
        """Generate a validated `Plan` from a fully-built prompt string.

        TODO:
          - call client.models.generate_content with the JSON schema config,
          - parse robustly (response.parsed -> fence-strip -> json.loads),
          - on parse failure, issue up to max_parse_repairs repair prompts.
        """
        raise NotImplementedError
