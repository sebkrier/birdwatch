"""Ask route for natural language Q&A about tweets."""

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

from app.services.ask import ask_about_tweets, smart_search, set_api_key

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


class AskRequest(BaseModel):
    question: str
    smart_mode: bool = True
    max_tweets: int = 100


class SetApiKeyRequest(BaseModel):
    api_key: str


@router.get("", response_class=HTMLResponse)
async def ask_page(request: Request):
    """Ask page for natural language questions."""
    return templates.TemplateResponse(
        "ask.html",
        {"request": request},
    )


@router.post("/query", response_class=JSONResponse)
async def ask_query(request: AskRequest):
    """Process a natural language question about tweets."""
    if not request.question.strip():
        return {"error": "Please enter a question"}

    if request.smart_mode:
        result = await smart_search(
            request.question,
            max_tweets=request.max_tweets,
        )
    else:
        result = await ask_about_tweets(
            request.question,
            max_tweets=request.max_tweets,
        )

    return result


@router.post("/set-api-key", response_class=JSONResponse)
async def set_gemini_api_key(request: SetApiKeyRequest):
    """Set the Gemini API key."""
    if not request.api_key.strip():
        return {"error": "API key cannot be empty"}

    set_api_key(request.api_key.strip())
    return {"success": True, "message": "API key set successfully"}
