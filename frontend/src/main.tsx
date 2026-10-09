import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

type Health={status:string;version:string;capabilities:Record<string,boolean>};
type Task={task_id:string;goal:string;state:string;dry_run:boolean;created_at:string};
const API=import.meta.env.VITE_API_URL||'http://127.0.0.1:8000';
function App(){const [health,setHealth]=useState<Health|null>(null);const [tasks,setTasks]=useState<Task[]>([]);const [goal,setGoal]=useState('');const [error,setError]=useState('');
 const refresh=async()=>{try{setHealth(await (await fetch(`${API}/api/v1/health`)).json());setTasks(await (await fetch(`${API}/api/v1/tasks`)).json())}catch(e){setError('Backend unavailable. Start visionos-api first.')}};
 useEffect(()=>{refresh();const id=setInterval(refresh,2000);return()=>clearInterval(id)},[]);
 const create=async()=>{if(!goal.trim())return;await fetch(`${API}/api/v1/tasks`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({goal,dry_run:true})});setGoal('');refresh()};
 return <main><header><div><p className="eyebrow">VISIONOS AI</p><h1>Computer-use control room</h1></div><span className={health?.status==='ok'?'pill ok':'pill'}>{health?.status||'offline'}</span></header><section className="grid"><article className="card"><h2>System health</h2><p>API version: {health?.version||'—'}</p>{health&&Object.entries(health.capabilities).map(([k,v])=><div className="cap" key={k}><span>{k}</span><b>{v?'available':'disabled'}</b></div>)}{error&&<p className="error">{error}</p>}</article><article className="card"><h2>Create task</h2><p className="muted">Tasks start in dry-run mode and require explicit desktop permission to control input.</p><div className="form"><input aria-label="Task goal" value={goal} onChange={e=>setGoal(e.target.value)} placeholder="Describe a task…"/><button onClick={create}>Create</button></div></article></section><section className="card"><div className="row"><h2>Task history</h2><button className="secondary" onClick={refresh}>Refresh</button></div>{tasks.length===0?<p className="muted">No persisted tasks yet.</p>:<div className="tasks">{tasks.map(t=><div className="task" key={t.task_id}><div><b>{t.goal}</b><small>{t.task_id}</small></div><span className="pill">{t.state}</span></div>)}</div>}</section></main>}
createRoot(document.getElementById('root')!).render(<App/>);
