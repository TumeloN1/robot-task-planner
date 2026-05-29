"""Robot Task Planner with self-correcting execution.

An LLM (Gemini) planner emits a JSON tool sequence; scripted MuJoCo primitives
execute it; temporal postcondition checkers verify each step; a diagnosis and
replanning loop self-corrects failures.
"""

__version__ = "0.1.0"
