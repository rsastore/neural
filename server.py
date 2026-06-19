"""Agent Server — REST API for Neural."""
import os, json, asyncio

_app = None
_session = None

def _get_session(provider, config):
    global _session
    if _session is None and provider:
        from agent import AgentSession
        _session = AgentSession(provider, config or {})
    return _session

def create_app(provider, config=None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    app = FastAPI(title="RSA Agentic API")
    # Optional API key authentication (set RSA_API_KEY env var)
    import os as _os2
    _API_KEY = _os2.environ.get("RSA_API_KEY", "")
    if _API_KEY:
        @app.middleware("http")
        async def auth_middleware(request, call_next):
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {_API_KEY}":
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=403, content={"error": "Invalid or missing API key"})
            return await call_next(request)

    _get_session(provider, config)

    @app.get("/status")
    async def status():
        from tools.builtin import list_tools
        from knowledge import knowledge_summary
        return {"status":"ok","tools":list_tools(),"knowledge":knowledge_summary()}

    @app.post("/chat")
    async def chat(body: dict):
        msg = body.get("message","")
        if not msg:
            raise HTTPException(400,"message required")
        async def gen():
            for ev in _session.run_stream(msg):
                yield json.dumps(ev)+"\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/chat/sync")
    async def chat_sync(body: dict):
        msg = body.get("message","")
        if not msg:
            raise HTTPException(400,"message required")
        import asyncio
        result = await asyncio.get_event_loop().run_in_executor(None, _session.run, msg)
        return {"response": result}

    @app.post("/plan")
    async def plan(body: dict):
        from planner import PlannerAgent
        p = PlannerAgent(_session)
        results = list(p.run_with_plan(body.get("goal","")))
        return {"results": results}

    @app.get("/knowledge")
    async def knowledge():
        from knowledge import get_facts, get_skills, knowledge_summary
        return {"summary":knowledge_summary(),"facts":get_facts()[-10:],"skills":get_skills()[-10:]}

    @app.post("/reset")
    async def reset():
        if _session: _session.reset()
        return {"status":"reset"}

    return app

def run_server(host="127.0.0.1", port=8765, provider=None, config=None):
    if host == "0.0.0.0":
        print("\033[33m⚠️  WARNING: Server exposed on ALL network interfaces!\033[0m")
        print("\033[33m   Anyone on the network can access the agent.\033[0m")
        print("\033[33m   Set RSA_API_KEY env var for authentication.\033[0m")
    import uvicorn
    app = create_app(provider, config)
    print(f"Neural API @ http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
