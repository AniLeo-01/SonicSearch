import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.dataset import load_manifest
from app.db import init_db, make_pool
from app import library
from app.embed import ColBERT, Embedder
from app.index import reindex
from app.search import Searcher, mmss


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # extensions + tables first
    app.state.pool = make_pool()  # then connections that register pgvector
    from sentence_transformers import CrossEncoder  # heavy import, only at startup

    reranker = CrossEncoder(settings.rerank_model, device="cpu")
    app.state.searcher = s = Searcher(app.state.pool, Embedder(settings.embed_model), reranker, ColBERT(settings.colbert_model))  # loads the models once
    with app.state.pool.connection() as conn:
        indexed = {r[0] for r in conn.execute("SELECT file_id FROM audio_files").fetchall()}
    if indexed != {e.file_id for e in load_manifest()}:  # a new database (e.g. a fresh Docker volume) or files changed outside the app
        print(reindex(s.emb, s.colbert))
    yield
    app.state.pool.close()


app = FastAPI(title="SonicSearch", lifespan=lifespan)


@app.get("/healthz")
def healthz(request: Request):
    with request.app.state.pool.connection() as conn:
        exts = [r[0] for r in conn.execute("SELECT extname FROM pg_extension ORDER BY extname").fetchall()]
    return {"status": "ok", "extensions": exts}


@app.get("/api/search")
def api_search(
    request: Request,
    q: str = Query(min_length=1, max_length=512),
    k: int = Query(10, ge=1, le=50),
    role: str | None = Query(None, pattern="^(host|guest)$"),
    mode: str = Query("hybrid", pattern="^(hybrid|lexical|semantic)$"),
):
    hits = request.app.state.searcher.search(q, k=k, role=role, mode=mode)
    return {"query": q, "hits": [{**asdict(h), "timestamp": mmss(h.start)} for h in hits]}


@app.get("/api/files")
def api_files(request: Request):
    with request.app.state.pool.connection() as conn:
        return {"files": library.listing(conn)}


@app.put("/api/files", status_code=202)
async def api_upload(
    request: Request,
    name: str = Query(min_length=1, max_length=255),  # the original file name: becomes the id and default title
    title: str | None = Query(None, max_length=200),
):
    """Raw audio in the body. Transcribing, speaker labelling and the re-index run in the background."""
    fd, path = tempfile.mkstemp(suffix=".upload")
    size = 0
    with open(fd, "wb") as f:
        async for part in request.stream():
            size += len(part)
            if size > library.MAX_UPLOAD:
                Path(path).unlink()
                raise HTTPException(413, f"larger than {library.MAX_UPLOAD // 2**20} MB")
            f.write(part)
    if not size:
        Path(path).unlink()
        raise HTTPException(400, "empty upload")
    s = request.app.state.searcher
    return {"file_id": library.add(Path(path), name, title, s.emb, s.colbert)}


@app.delete("/api/files/{file_id}", status_code=202)
def api_remove(request: Request, file_id: str):
    """Leaves search after the background re-index; its files move to data/removed/<file_id>/."""
    s = request.app.state.searcher
    try:
        library.remove(file_id, s.emb, s.colbert)
    except LookupError:
        raise HTTPException(404, "no such recording") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"file_id": file_id}


# after the API routes, so "/" doesn't shadow them; StaticFiles answers Range requests, which audio seeking needs
app.mount("/audio", StaticFiles(directory=settings.data_dir / "audio"), name="audio")
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")