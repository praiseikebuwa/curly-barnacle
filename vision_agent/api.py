from __future__ import annotations
import os, uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from .events import EventBus
from .ingestion import ImageIngestor
from .memory import MemoryStore
from .perception import ScreenObserver
from .schemas import HealthResponse, TaskCreate, TaskRecord, TaskState
from .state import transition

VERSION = "0.3.0"
class ImageURLRequest(BaseModel): url: str = Field(min_length=1, max_length=2048)

def create_api() -> FastAPI:
    app = FastAPI(title="VisionOS AI", version=VERSION)
    bus = EventBus(); tasks: dict[str, TaskRecord] = {}; memory = MemoryStore(os.getenv("VISION_AGENT_DB", "vision_agent.db")); observer = ScreenObserver()
    app.state.event_bus = bus; app.state.tasks = tasks; app.state.memory = memory
    @app.get("/api/v1/health", response_model=HealthResponse)
    def health():
        llm_configured = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_BASE"))
        return HealthResponse(status="ok", version=VERSION, capabilities={"screenshot": True, "ocr": True, "desktop_control": False, "multimodal_model": llm_configured})
    @app.post("/api/v1/tasks", response_model=TaskRecord, status_code=201)
    def create_task(request: TaskCreate):
        now = datetime.now(timezone.utc); task = TaskRecord(task_id=str(uuid.uuid4()), goal=request.goal, state=TaskState.created, created_at=now, updated_at=now, dry_run=request.dry_run, max_actions=request.max_actions)
        tasks[task.task_id] = task; bus.publish("task_created", "Task created", task_id=task.task_id); return task
    @app.get("/api/v1/tasks", response_model=list[TaskRecord])
    def list_tasks(): return list(tasks.values())
    @app.get("/api/v1/tasks/{task_id}", response_model=TaskRecord)
    def get_task(task_id: str):
        if task_id not in tasks: raise HTTPException(404, "Task not found")
        return tasks[task_id]
    @app.post("/api/v1/tasks/{task_id}/cancel", response_model=TaskRecord)
    def cancel_task(task_id: str):
        if task_id not in tasks: raise HTTPException(404, "Task not found")
        task = tasks[task_id]
        try: task.state = transition(task.state, TaskState.cancelled)
        except ValueError as exc: raise HTTPException(409, str(exc)) from exc
        task.updated_at = datetime.now(timezone.utc); bus.publish("task_cancelled", "Task cancelled", task_id=task_id); return task
    @app.post("/api/v1/observations/capture")
    def capture_observation(ocr: bool = False):
        try: observation = observer.capture(use_ocr=ocr); record_id = memory.save_observation(observation)
        except Exception as exc: raise HTTPException(503, f"Screen capture unavailable: {exc}") from exc
        bus.publish("observation_created", "Authorized screenshot captured", observation_id=record_id, image_path=observation.image_path)
        return {"observation_id": record_id, "timestamp": observation.timestamp, "width": observation.width, "height": observation.height, "image_path": observation.image_path, "elements": [e.__dict__ for e in observation.elements]}
    @app.post("/api/v1/images/url")
    def import_image_url(request: ImageURLRequest):
        try: path = ImageIngestor().import_url(request.url)
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc
        except Exception as exc: raise HTTPException(502, "Image download failed safely") from exc
        return {"path": str(path), "source": "https", "status": "imported"}
    @app.get("/api/v1/events")
    def events(): return bus.recent()
    return app
