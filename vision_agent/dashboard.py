from __future__ import annotations

import json
from flask import Flask, Response, jsonify

from .events import EventBus

HTML = """<!doctype html><html><head><meta charset='utf-8'><title>Vision Agent Monitor</title><style>body{font:15px system-ui;background:#111;color:#eee;margin:2rem}h1{color:#7dd3fc}.event{border:1px solid #333;border-radius:8px;padding:.7rem;margin:.5rem 0}.type{color:#86efac;font-weight:700}.time{color:#999;font-size:.8rem}</style></head><body><h1>Vision Agent Monitor</h1><p>Live execution events and memory telemetry.</p><div id='events'></div><script>const box=document.querySelector('#events');function render(es){box.innerHTML=es.slice().reverse().map(e=>`<div class='event'><span class='type'>${e.type}</span> ${e.message}<div class='time'>${e.timestamp}</div><pre>${JSON.stringify(e.data,null,2)}</pre></div>`).join('')}async function refresh(){render(await (await fetch('/api/events')).json())}refresh();setInterval(refresh,1000)</script></body></html>"""


def create_app(event_bus: EventBus | None = None) -> Flask:
    bus = event_bus or EventBus()
    app = Flask(__name__)
    app.config["EVENT_BUS"] = bus

    @app.get("/")
    def index() -> str:
        return HTML

    @app.get("/api/events")
    def events():
        return jsonify(bus.recent())

    @app.get("/api/state")
    def state():
        items = bus.recent()
        return jsonify({"event_count": len(items), "last_event": items[-1] if items else None})

    @app.get("/events")
    def stream():
        def generate():
            yield "data: " + json.dumps(bus.recent()) + "\n\n"
        return Response(generate(), mimetype="text/event-stream")

    return app
