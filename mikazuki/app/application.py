import asyncio
import mimetypes
import os
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from mikazuki.app.config import app_config
from mikazuki.app.api import load_schemas, load_presets
from mikazuki.app.api import router as api_router
from mikazuki.app.proxy import router as proxy_router
from mikazuki.app.training_pages import (
    patch_frontend_app_js,
    virtual_asset,
    virtual_page_paths,
)
from mikazuki.utils.devices import check_torch_gpu

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

FRONTEND_DIST_DIR = Path("./frontend/dist")
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            raise ex


def _safe_frontend_path(base_dir: Path, relative_path: str) -> Path:
    """Resolve a frontend path while preventing traversal outside the dist directory."""
    base = base_dir.resolve()
    target = (base / relative_path).resolve()
    if target != base and base not in target.parents:
        raise HTTPException(status_code=404)
    return target


async def app_startup():
    app_config.load_config()
    await load_schemas()
    await load_presets()
    await asyncio.to_thread(check_torch_gpu)

    if sys.platform == "win32" and os.environ.get("MIKAZUKI_DEV", "0") != "1":
        webbrowser.open(f'http://{os.environ["MIKAZUKI_HOST"]}:{os.environ["MIKAZUKI_PORT"]}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_startup()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(proxy_router)

cors_config = os.environ.get("MIKAZUKI_APP_CORS", "")
if cors_config != "":
    if cors_config == "1":
        cors_config = ["http://localhost:8004", "*"]
    else:
        cors_config = cors_config.split(";")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    return response


app.include_router(api_router, prefix="/api")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIST_DIR / "index.html")


@app.get("/favicon.ico", response_class=FileResponse)
async def favicon():
    return FileResponse("assets/favicon.ico")


@app.get("/assets/{asset_name:path}")
async def frontend_asset(asset_name: str):
    """Serve the pinned frontend plus runtime-injected backend training pages."""
    generated = virtual_asset(asset_name)
    if generated is not None:
        return Response(content=generated, media_type="application/javascript")

    asset_path = _safe_frontend_path(FRONTEND_ASSETS_DIR, asset_name)
    if not asset_path.is_file():
        raise HTTPException(status_code=404)

    if asset_name == "app.547295de.js":
        content = asset_path.read_text(encoding="utf-8")
        content = patch_frontend_app_js(content)
        return Response(content=content, media_type="application/javascript")

    return FileResponse(asset_path)


def _virtual_training_shell():
    """All runtime VuePress pages use the same SPA shell; route data is injected in app.js."""
    index_path = FRONTEND_DIST_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(index_path)


# Direct browser navigation to generated .html routes must reach the VuePress
# shell before the fallback StaticFiles mount handles the request.
@app.get("/lora/chroma.html")
@app.get("/lora/anima.html")
@app.get("/finetune/sdxl.html")
@app.get("/finetune/flux.html")
@app.get("/finetune/anima.html")
async def virtual_training_page():
    return _virtual_training_shell()


# Keep the path list imported and checked at startup/module import so adding a
# new virtual page without adding a direct .html route is caught by tests and
# is visible to maintainers here.
VIRTUAL_PAGE_PATHS = virtual_page_paths()


app.mount("/", SPAStaticFiles(directory="frontend/dist", html=True), name="static")
