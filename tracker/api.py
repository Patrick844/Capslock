"""FastAPI surface: POST /digest."""
from fastapi import FastAPI, HTTPException

from tracker.agent import run_digest
from tracker.schemas import DigestRequest, DigestResponse

app = FastAPI(title="CapsLock AI Industry Mini-Tracker")


@app.post("/digest", response_model=DigestResponse)
async def create_digest(request: DigestRequest) -> DigestResponse:
    try:
        return await run_digest(request)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
