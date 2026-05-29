import rtp.primitives.checks  # noqa: F401  (registers checkers)
import rtp.primitives.manipulation  # noqa: F401
import rtp.primitives.motion  # noqa: F401
import rtp.primitives.queries  # noqa: F401
from rtp.primitives.registry import PRIMITIVES

EXPECTED_TOOLS = {
    "find_object",
    "look_around",
    "move_to_pregrasp",
    "move_to_pose",
    "grasp",
    "release",
    "check_grasp_success",
    "check_task_success",
}


def test_all_expected_tools_registered():
    assert EXPECTED_TOOLS.issubset(set(PRIMITIVES))


def test_check_primitives_flagged_as_checks():
    assert PRIMITIVES["check_grasp_success"].is_check is True
    assert PRIMITIVES["grasp"].is_check is False
