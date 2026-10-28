"""FastAPI app for the TAORG dashboard: a JSON API over saved runs plus the static UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from taorg.dashboard import explorer, runs

STATIC = Path(__file__).parent / "static"
DOWNLOADABLE = {"metrics.json", "metadata.json", "config.resolved.yaml", "results.json",
                "timing.json", "INTERRUPTED.txt", "STOPPED.txt"}


class SimulateRequest(BaseModel):
    run_id: str
    condition: str
    seed: int
    video_index: int = Field(ge=0)
    methods: list[str]


def create_app(runs_root: Path) -> FastAPI:
    runs_root = Path(runs_root).resolve()
    app = FastAPI(title="TAORG dashboard", docs_url="/api/docs", redoc_url=None)

    def _guard(fn, *args):
        try:
            return fn(*args)
        except runs.RunNotFound as exc:
            raise HTTPException(404, f"run not found or not usable here: {exc}") from exc
        except explorer.ExplorerError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "runs_root": str(runs_root), "runs_root_exists": runs_root.is_dir()}

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return runs.overview(runs_root)

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return runs.list_runs(runs_root)

    @app.get("/api/matrix/{run_id}")
    def matrix(run_id: str) -> dict[str, Any]:
        return _guard(runs.matrix_summary, runs_root, run_id)

    @app.get("/api/complexity/{run_id:path}")
    def complexity(run_id: str) -> dict[str, Any]:
        info, path = _guard(runs.run_path, runs_root, run_id)
        if info["kind"] != "complexity":
            raise HTTPException(404, "not a complexity run")
        return {"run": info, **runs.read_json(path / "results.json")}

    @app.get("/api/recovery/{run_id:path}")
    def recovery(run_id: str) -> dict[str, Any]:
        info, path = _guard(runs.run_path, runs_root, run_id)
        if info["kind"] != "param_recovery":
            raise HTTPException(404, "not a parameter-recovery run")
        return {"run": info, **runs.read_json(path / "results.json")}

    @app.get("/api/files/{run_id:path}/{name}")
    def download(run_id: str, name: str) -> FileResponse:
        info, path = _guard(runs.run_path, runs_root, run_id)
        if name not in DOWNLOADABLE or name not in info["files"] or not path.is_dir():
            raise HTTPException(404, "file not available")
        return FileResponse(path / name, filename=f"{run_id.replace('/', '_')}_{name}")

    @app.get("/api/explorer/{run_id}/options")
    def explorer_options(run_id: str) -> dict[str, Any]:
        return _guard(explorer.options, runs_root, run_id)

    @app.post("/api/explorer/simulate")
    def simulate(req: SimulateRequest) -> dict[str, Any]:
        return _guard(explorer.simulate, runs_root, req.run_id, req.condition, req.seed,
                      req.video_index, req.methods)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
