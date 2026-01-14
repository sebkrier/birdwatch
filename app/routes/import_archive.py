"""Archive import routes."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator
import uuid

from fastapi import APIRouter, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.db.pool import get_cursor, get_connection
from app.db import queries

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

# In-memory job tracking (in production, use Redis or similar)
import_jobs: dict[str, dict] = {}


@router.get("", response_class=HTMLResponse)
async def import_page(request: Request):
    """Import archive page."""
    async with get_cursor() as cur:
        # Get import history
        await cur.execute(
            """
            SELECT * FROM import_history
            ORDER BY started_at DESC
            LIMIT 10
            """
        )
        history = await cur.fetchall()

        # Get current stats
        tweet_count = await queries.get_tweet_count(cur)

    return templates.TemplateResponse(
        "import.html",
        {"request": request, "history": history, "tweet_count": tweet_count},
    )


@router.post("")
async def upload_archive(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload and process a Twitter archive."""
    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file temporarily
    temp_path = Path(f"/tmp/birdwatch_import_{job_id}.zip")
    content = await file.read()
    temp_path.write_bytes(content)

    # Initialize job tracking
    import_jobs[job_id] = {
        "status": "pending",
        "tweets_processed": 0,
        "tweets_total": 0,
        "media_processed": 0,
        "current_file": None,
        "error": None,
    }

    # Start background import task
    background_tasks.add_task(run_import, job_id, temp_path, file.filename)

    # Return page with progress tracker
    return templates.TemplateResponse(
        "import_progress.html",
        {"request": request, "job_id": job_id},
    )


async def run_import(job_id: str, archive_path: Path, filename: str):
    """Background task to run the import."""
    from app.services.importer import import_archive

    try:
        import_jobs[job_id]["status"] = "in_progress"

        async for progress in import_archive(archive_path, filename):
            import_jobs[job_id].update(progress)

        import_jobs[job_id]["status"] = "completed"
    except Exception as e:
        import_jobs[job_id]["status"] = "failed"
        import_jobs[job_id]["error"] = str(e)
    finally:
        # Clean up temp file
        if archive_path.exists():
            archive_path.unlink()


@router.get("/progress/{job_id}")
async def import_progress(job_id: str):
    """SSE endpoint for import progress updates."""

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            if job_id not in import_jobs:
                yield f"data: {{'error': 'Job not found'}}\n\n"
                break

            job = import_jobs[job_id]
            yield f"data: {job}\n\n"

            if job["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
