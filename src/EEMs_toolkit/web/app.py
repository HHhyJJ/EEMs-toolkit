from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .routers import analysis, data, parafac, preprocess
from .session import SESSION_TTL, STORE, get_session  # noqa: F401 - get_session is re-exported for routers

STATIC_DIR = Path(__file__).parent / 'static'
COOKIE_NAME = 'eems_sid'

_MISSING_STATIC = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>EEMs-toolkit Web</title></head>
<body style="font-family: system-ui, sans-serif; margin: 3rem; line-height: 1.7">
<h1>前端资源缺失</h1>
<p>未找到 <code>EEMs_toolkit/web/static/index.html</code>。</p>
<p>请在仓库的 <code>frontend</code> 目录中执行：</p>
<pre style="background:#f4f4f4; padding: 1rem">npm install
npm run build</pre>
<p>然后重新启动 <code>eems-web</code>。</p>
</body></html>"""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    yield
    STORE.close_all()


def create_app() -> FastAPI:
    app = FastAPI(title='EEMs-toolkit Web', version='0.3.0', lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.middleware('http')
    async def attach_session(request: Request, call_next):
        if request.method == 'OPTIONS' or not request.url.path.startswith('/api/'):
            return await call_next(request)
        sid = request.cookies.get(COOKIE_NAME)
        session = STORE.get(sid)
        new_sid: Optional[str] = None
        if session is None:
            session = STORE.get_or_create()
            new_sid = session.id
        request.state.session = session
        response = await call_next(request)
        if new_sid is not None:
            response.set_cookie(
                COOKIE_NAME, new_sid, max_age=SESSION_TTL,
                httponly=True, samesite='lax',
            )
        return response

    app.include_router(data.router)
    app.include_router(preprocess.router)
    app.include_router(parafac.router)
    app.include_router(analysis.router)

    if STATIC_DIR.is_dir():
        app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

    @app.get('/', response_class=HTMLResponse)
    def index():
        index_file = STATIC_DIR / 'index.html'
        if index_file.is_file():
            return index_file.read_text(encoding='utf-8')
        return _MISSING_STATIC

    @app.get('/{full_path:path}', response_class=HTMLResponse)
    def spa(full_path: str):
        if full_path.startswith('api/'):
            return JSONResponse({'detail': 'Not found'}, status_code=404)
        index_file = STATIC_DIR / 'index.html'
        if index_file.is_file():
            return index_file.read_text(encoding='utf-8')
        return _MISSING_STATIC

    return app


app = create_app()
