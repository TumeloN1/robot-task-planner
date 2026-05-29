import numpy as np

from rtp.perception.api import ObjectState, Pose, SceneState


def _obj(oid, name, category, visible=True):
    return ObjectState(
        id=oid,
        name=name,
        category=category,
        color=name.split()[0] if " " in name else None,
        pose=Pose(position=np.zeros(3), quaternion=np.array([1.0, 0, 0, 0])),
        aabb=np.array([[-0.02, -0.02, 0.0], [0.02, 0.02, 0.1]]),
        visible=visible,
    )


def test_graspable_category_rules():
    assert _obj("o1", "red mug", "mug").graspable is True
    assert _obj("t1", "tray", "tray").graspable is False
    assert _obj("tab", "table", "table").graspable is False


def test_scene_state_find_by_name_and_id():
    mug = _obj("o1", "red mug", "mug")
    scene = SceneState(step=0, visible_objects=[mug])
    assert scene.find("red mug") is mug
    assert scene.find("o1") is mug
    assert scene.find("blue block") is None


def test_all_known_prefers_visible_over_remembered():
    visible = _obj("o1", "red mug", "mug", visible=True)
    remembered = _obj("o2", "blue block", "block", visible=False)
    scene = SceneState(step=1, visible_objects=[visible], remembered_objects=[remembered])
    known = scene.all_known()
    assert {o.id for o in known} == {"o1", "o2"}
