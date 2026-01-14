"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.db.pool import init_pool, close_pool, get_cursor
from app.db import queries
from app.routes import search, tweet, explore, import_archive, ask


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # Startup
    await init_pool()
    yield
    # Shutdown
    await close_pool()


app = FastAPI(
    title="Birdwatch",
    description="Personal Twitter Archive & Externalized Memory",
    version="0.1.0",
    lifespan=lifespan,
)

# Templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Media files
settings = get_settings()
media_dir = settings.media_dir
if media_dir.exists():
    app.mount("/media", StaticFiles(directory=str(media_dir)), name="media")

# Include routers
app.include_router(search.router, prefix="", tags=["search"])
app.include_router(tweet.router, prefix="/tweet", tags=["tweets"])
app.include_router(explore.router, prefix="/explore", tags=["explore"])
app.include_router(import_archive.router, prefix="/import", tags=["import"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page with search."""
    async with get_cursor() as cur:
        await cur.execute('''
            SELECT c.id, c.name, c.color, COUNT(tc.tweet_id) as tweet_count
            FROM categories c
            LEFT JOIN tweet_categories tc ON c.id = tc.category_id
            GROUP BY c.id, c.name, c.color
            ORDER BY tweet_count DESC
        ''')
        categories = await cur.fetchall()

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "categories": categories}
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
