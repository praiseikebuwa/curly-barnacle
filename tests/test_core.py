from pathlib import Path

from vision_agent.controls import ActionExecutor
from vision_agent.memory import MemoryStore
from vision_agent.models import Action, Element, Observation
from vision_agent.planner import Planner


def test_element_center():
    assert Element("Save", x=10, y=20, width=80, height=40).center == (50, 40)


def test_planner_creates_safe_text_actions():
    plan = Planner().plan("search for Python tutorials")
    assert [a.kind for a in plan.actions] == ["type", "press"]
    assert plan.actions[0].args["text"] == "Python tutorials"


def test_executor_dry_run_does_not_require_pyautogui():
    result = ActionExecutor(dry_run=True).execute(Action("type", {"text": "hello"}))
    assert result.startswith("dry-run")


def test_memory_persists_actions(tmp_path: Path):
    store = MemoryStore(tmp_path / "agent.db")
    store.save_action(Action("wait", {"seconds": 1}), True, "dry-run")
    store.save_workflow("wait", [Action("wait", {"seconds": 1})])
    assert len(store.recent_workflows()) == 1
    store.close()


def test_observation_serializes_elements():
    observation = Observation("now", 100, 200, elements=(Element("OK"),))
    assert '"text": "OK"' in observation.to_json()
