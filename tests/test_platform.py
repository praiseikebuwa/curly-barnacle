from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vision_agent.api import create_api
from vision_agent.dataset_lab import Annotation, export_coco, split_examples
from vision_agent.ingestion import ImageIngestor
from vision_agent.safety import SafetyPolicy
from vision_agent.schemas import ActionRequest, ActionType, BoundingBox
from vision_agent.state import transition
from vision_agent.schemas import TaskState

def test_api_task_lifecycle():
 c=TestClient(create_api()); assert c.get('/api/v1/health').json()['capabilities']['external_ai_providers'] is False; created=c.post('/api/v1/tasks',json={'goal':'inspect screen','dry_run':True}).json(); assert created['state']=='CREATED'; assert c.post(f"/api/v1/tasks/{created['task_id']}/cancel").json()['state']=='CANCELLED'
def test_action_schema_blocks_code():
 with pytest.raises(ValidationError): ActionRequest(action_id='x',action_type=ActionType.wait,parameters={'shell':'rm -rf /'})
def test_state_transitions_and_safety():
 assert transition(TaskState.created,TaskState.validating)==TaskState.validating
 with pytest.raises(ValueError): transition(TaskState.completed,TaskState.running)
 with pytest.raises(PermissionError): SafetyPolicy().check('type_text',{},0)
def test_dataset_coco_and_split(tmp_path:Path):
 ann=Annotation('a1','img1','button',BoundingBox(x=1,y=2,width=10,height=12),approved=True); out=tmp_path/'coco.json'; export_coco([{'id':'img1','file_name':'x.png','width':20,'height':20}],[ann],out); assert 'categories' in out.read_text(); assert sum(map(len,split_examples([{'id':str(i)} for i in range(10)]).values()))==10
def test_ingestor_rejects_private_and_missing_files(tmp_path:Path):
 ing=ImageIngestor(tmp_path)
 with pytest.raises(ValueError): ing.import_url('http://localhost/a.png')
 with pytest.raises(ValueError): ing.import_local(tmp_path/'missing.png')
