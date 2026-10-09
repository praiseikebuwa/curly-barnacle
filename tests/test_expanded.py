from pathlib import Path
from vision_agent.controls import ActionExecutor
from vision_agent.memory import MemoryStore
from vision_agent.models import Action, Element, Observation
from vision_agent.planner import Planner
from vision_agent.recovery import RetryPolicy, RetryRunner
from vision_agent.dashboard import create_app
from vision_agent.events import EventBus

def test_element_center(): assert Element('Save',x=10,y=20,width=80,height=40).center==(50,40)
def test_planner_creates_safe_text_actions():
 p=Planner().plan('search for Python tutorials'); assert [a.kind for a in p.actions]==['type','press']; assert p.actions[0].args['text']=='Python tutorials'
def test_executor_dry_run_does_not_require_pyautogui(): assert ActionExecutor(dry_run=True).execute(Action('type',{'text':'hello'})).startswith('dry-run')
def test_memory_persists_actions(tmp_path:Path):
 s=MemoryStore(tmp_path/'agent.db'); s.save_action(Action('wait',{'seconds':1}),True,'dry-run'); s.save_workflow('wait',[Action('wait',{'seconds':1})]); assert len(s.recent_workflows())==1; s.close()
def test_observation_serializes_elements(): assert '"text": "OK"' in Observation('now',100,200,elements=(Element('OK'),)).to_json()
def test_retry_recovers():
 calls=[]
 def flaky():
  calls.append(1)
  if len(calls)<3: raise OSError('temporary')
  return 'ok'
 assert RetryRunner(RetryPolicy(max_attempts=3,initial_delay=0),sleep=lambda _:None).run('flaky',flaky)=='ok'; assert len(calls)==3
def test_dashboard_api():
 bus=EventBus(); bus.publish('test','hello',count=1); c=create_app(bus).test_client(); assert c.get('/').status_code==200; assert c.get('/api/events').get_json()[0]['message']=='hello'
