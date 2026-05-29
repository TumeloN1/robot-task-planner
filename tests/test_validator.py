from rtp.planner.schema import Plan, ToolCall, build_response_json_schema
from rtp.planner.validator import ValidationResult


def test_plan_requires_rationale_field():
    schema = build_response_json_schema()
    assert "rationale" in schema["properties"]
    assert "rationale" in schema.get("required", [])


def test_validation_result_repair_text_is_concise():
    res = ValidationResult(valid=False, errors=["unknown tool 'fly'", "object 'ufo' not found"])
    text = res.as_repair_text()
    assert "fly" in text and "ufo" in text


def test_plan_roundtrip():
    plan = Plan(
        rationale="initial",
        tool_calls=[ToolCall(tool="find_object", args={"query": "red mug"})],
    )
    assert plan.tool_calls[0].tool == "find_object"
