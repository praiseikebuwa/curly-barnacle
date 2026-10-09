from __future__ import annotations
import uuid
from datetime import datetime,timezone
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from .events import EventBus
from .ingestion import ImageIngestor
from .memory import MemoryStore
from .model_status import vision_status
from .perception import ScreenObserver
from .schemas import HealthResponse,TaskCreate,TaskRecord,TaskState
from .state import transition
VERSION='0.5.0'
class ImageURLRequest(BaseModel): url:str=Field(min_length=1,max_length=2048)
def create_api()->FastAPI:
 app=FastAPI(title='Vision Agent',version=VERSION);bus=EventBus();tasks={};memory=MemoryStore();observer=ScreenObserver();app.state.event_bus=bus;app.state.tasks=tasks
 @app.get('/api/v1/health',response_model=HealthResponse)
 def health(): return HealthResponse(status='ok',version=VERSION,capabilities={'screenshot':True,'ocr':True,'desktop_control':False,'from_scratch_vision_training':True,'external_ai_providers':False})
 @app.get('/api/v1/models/vision/status')
 def model_status(): return vision_status()
 @app.post('/api/v1/tasks',response_model=TaskRecord,status_code=201)
 def create_task(request:TaskCreate):
  now=datetime.now(timezone.utc);task=TaskRecord(task_id=str(uuid.uuid4()),goal=request.goal,state=TaskState.created,created_at=now,updated_at=now,dry_run=request.dry_run,max_actions=request.max_actions);tasks[task.task_id]=task;bus.publish('task_created','Task created',task_id=task.task_id);return task
 @app.get('/api/v1/tasks',response_model=list[TaskRecord])
 def list_tasks(): return list(tasks.values())
 @app.get('/api/v1/tasks/{task_id}',response_model=TaskRecord)
 def get_task(task_id:str):
  if task_id not in tasks: raise HTTPException(404,'Task not found')
  return tasks[task_id]
 @app.post('/api/v1/tasks/{task_id}/cancel',response_model=TaskRecord)
 def cancel_task(task_id:str):
  if task_id not in tasks: raise HTTPException(404,'Task not found')
  try:tasks[task_id].state=transition(tasks[task_id].state,TaskState.cancelled)
  except ValueError as exc:raise HTTPException(409,str(exc)) from exc
  tasks[task_id].updated_at=datetime.now(timezone.utc);return tasks[task_id]
 @app.post('/api/v1/observations/capture')
 def capture_observation(ocr:bool=False):
  try:o=observer.capture(use_ocr=ocr);rid=memory.save_observation(o)
  except Exception as exc:raise HTTPException(503,f'Screen capture unavailable: {exc}') from exc
  return {'observation_id':rid,'timestamp':o.timestamp,'width':o.width,'height':o.height,'image_path':o.image_path,'elements':[e.__dict__ for e in o.elements]}
 @app.post('/api/v1/images/url')
 def import_image_url(request:ImageURLRequest):
  try:path=ImageIngestor().import_url(request.url)
  except ValueError as exc:raise HTTPException(400,str(exc)) from exc
  except Exception as exc:raise HTTPException(502,'Image download failed safely') from exc
  return {'path':str(path),'source':'https','status':'imported'}
 @app.get('/api/v1/events')
 def events():return bus.recent()
 return app
