import json
from pathlib import Path

from PIL import Image

from vision_agent.dashboard import create_app
from vision_agent.dataset import load_dataset
from vision_agent.events import EventBus
from vision_agent.llm import MultimodalReasoner
from vision_agent.recovery import RetryPolicy, RetryRunner
from vision_agent.training import IntentClassifier


def test_dataset_trains_and_round_trips(tmp_path: Path):
    examples = load_dataset("data/interface_actions.jsonl")
    model = IntentClassifier.train(examples)
    label, confidence = model.predict("please type hello")
    assert label == "type"
    assert confidence >= 0
    path = tmp_path / "model.json"
    model.save(path)
    assert IntentClassifier.load(path).predict("press enter")[0] == "press"


def test_retry_recovers():
    calls = []
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("temporary")
        return "ok"
    runner = RetryRunner(RetryPolicy(max_attempts=3, initial_delay=0), sleep=lambda _: None)
    assert runner.run("flaky", flaky) == "ok"
    assert len(calls) == 3


def test_dashboard_api():
    bus = EventBus()
    bus.publish("test", "hello", count=1)
    client = create_app(bus).test_client()
    assert client.get("/").status_code == 200
    payload = client.get("/api/events").get_json()
    assert payload[0]["message"] == "hello"
    assert client.get("/api/state").get_json()["event_count"] == 1


def test_multimodal_request_shape(tmp_path: Path):
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (2, 2), "white").save(image_path)
    class Message: content = json.dumps({"actions": [], "notes": "safe"})
    class Choice: message = Message()
    class Response: choices = [Choice()]
    class Completions:
        def create(self, **kwargs):
            assert kwargs["messages"][1]["content"][1]["type"] == "image_url"
            assert kwargs["response_format"]["json_schema"]["strict"] is True
            return Response()
    class Client:
        chat = type("Chat", (), {"completions": Completions()})()
    from vision_agent.models import Observation
    plan = MultimodalReasoner(client=Client()).plan("inspect", Observation("now", 2, 2, str(image_path)))
    assert plan.notes == "safe"
