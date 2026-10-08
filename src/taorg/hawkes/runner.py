"""Checkpointed, bounded-parallel job execution for Phase 5H-1 (amendment A1, technical only).

Guarantees:
- Every unit of work (a "task", e.g. one fit) has a deterministic id = sha256 of its canonical
  spec (configuration, data set, seed, code hash). Its checkpoint is written atomically
  (tmp file + os.replace) and is reused only if the stored spec equals the requested spec.
- A job is a group of tasks run in one worker. Jobs are idempotent: completed tasks are skipped
  on resume. Jobs declare dependencies; a job whose dependency failed is recorded as BLOCKED.
- Failures are caught per job, recorded with traceback in attempts.jsonl (append-only, never
  rewritten) and do not stop other jobs. Re-running a stage retries only incomplete work.
- Duplicate task ids in a manifest raise; completeness is checked against the manifest.
- Scheduling never touches seeds or data: task results depend only on their spec.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
import traceback
import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

CODE_FILES = ("glm.py", "vidor.py", "nulls1.py", "phase5h1.py", "phase5h1_run.py", "hyp1.py")


def code_hash() -> str:
    """Hash of the Phase 5H-1 computational modules (part of every task spec)."""
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in CODE_FILES:
        p = here / f
        if p.exists():
            h.update(f.encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def spec_id(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:20]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class Checkpoints:
    """Task checkpoint store: <root>/<task_id>.json (+ optional <task_id>.npz)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, tid: str) -> Path:
        return self.root / f"{tid}.json"

    def get(self, spec: dict[str, Any]) -> dict[str, Any] | None:
        tid = spec_id(spec)
        p = self.path(tid)
        if not p.exists():
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("spec") != json.loads(json.dumps(spec, sort_keys=True, default=str)):
            raise RuntimeError(f"checkpoint {p} has a different spec (hash collision or tampering)")
        return rec if rec.get("status") == "complete" else None

    def put(self, spec: dict[str, Any], payload: dict[str, Any], arrays: dict[str, Any] | None = None) -> str:
        tid = spec_id(spec)
        if arrays is not None:
            atomic_savez(self.root / f"{tid}.npz", **arrays)
        rec = {"task_id": tid, "spec": json.loads(json.dumps(spec, sort_keys=True, default=str)),
               "status": "complete", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "host": platform.node(), "pid": os.getpid(), **payload}
        atomic_write_text(self.path(tid), json.dumps(rec, indent=1, default=float))
        return tid

    def arrays(self, tid: str) -> Any:
        import numpy as np
        return np.load(self.root / f"{tid}.npz")


@dataclass
class Job:
    """One schedulable unit: fn(**kwargs) runs in a worker process and returns a JSON-able summary."""

    jid: str
    fn: Callable[..., Any]
    kwargs: dict[str, Any]
    deps: list[str] = field(default_factory=list)
    weight: int = 1                          # memory slots this job occupies


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:                                     # keep the desktop responsive
        import psutil
        p = psutil.Process()
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception:                        # noqa: BLE001 - priority is best effort
        pass


def _call(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    import psutil
    t0 = time.time()
    try:
        out = fn(**kwargs)
        return {"status": "complete", "result": out, "elapsed_s": time.time() - t0,
                "peak_rss_gb": psutil.Process().memory_info().peak_wset / 1e9
                if hasattr(psutil.Process().memory_info(), "peak_wset") else None}
    except Exception as e:                   # noqa: BLE001 - every failure is recorded, not hidden
        return {"status": "failed", "error": repr(e), "traceback": traceback.format_exc(),
                "elapsed_s": time.time() - t0}


def run_jobs(jobs: list[Job], workers: int, log_path: Path, slots: int | None = None) -> dict[str, str]:
    """Run jobs with dependencies on a bounded process pool. Returns {jid: status}.

    workers: max concurrent processes; slots: memory budget in job-weight units (default = workers).
    workers == 0 runs serially in-process (reference mode for serial/parallel agreement checks).
    """
    ids = [j.jid for j in jobs]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate job ids: {dup[:5]}")
    byid = {j.jid: j for j in jobs}
    for j in jobs:
        for d in j.deps:
            if d not in byid:
                raise ValueError(f"job {j.jid} depends on unknown job {d}")
    slots = slots or max(workers, 1)
    status: dict[str, str] = {}
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(rec: dict[str, Any]) -> None:
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def ready(j: Job) -> bool:
        return all(status.get(d) == "complete" for d in j.deps)

    def blocked(j: Job) -> bool:
        return any(status.get(d) in ("failed", "blocked") for d in j.deps)

    pending = list(jobs)
    if workers == 0:
        while pending:
            progressed = False
            for j in list(pending):
                if blocked(j):
                    status[j.jid] = "blocked"
                    log({"job": j.jid, "event": "blocked"})
                    pending.remove(j)
                    progressed = True
                elif ready(j):
                    log({"job": j.jid, "event": "start", "mode": "serial"})
                    r = _call(j.fn, j.kwargs)
                    status[j.jid] = r["status"]
                    log({"job": j.jid, "event": "end", **r})
                    pending.remove(j)
                    progressed = True
            if not progressed:
                raise RuntimeError("dependency cycle")
        return status

    ctx = mp.get_context("spawn")             # one fresh process per job: memory is returned on exit
    running: dict[Any, tuple[Job, Any, Any]] = {}
    used = 0
    while pending or running:
        for j in list(pending):
            if blocked(j):
                status[j.jid] = "blocked"
                log({"job": j.jid, "event": "blocked"})
                pending.remove(j)
            elif ready(j) and len(running) < workers and (used + j.weight <= slots or not running):
                rd, wr = ctx.Pipe(duplex=False)
                pr = ctx.Process(target=_proc_entry, args=(j.fn, j.kwargs, wr), daemon=False)
                pr.start()
                wr.close()
                running[pr.sentinel] = (j, pr, rd)
                used += j.weight
                pending.remove(j)
                log({"job": j.jid, "event": "start", "mode": f"spawn{workers}", "pid": pr.pid})
        if not running:
            if pending:
                raise RuntimeError("no runnable job (dependency cycle?)")
            break
        for sent in mp_wait(list(running)):
            j, pr, rd = running.pop(sent)
            used -= j.weight
            r = None
            try:
                if rd.poll():
                    r = rd.recv()
            except (EOFError, OSError):
                r = None
            pr.join()
            if r is None:                        # crashed without a result (e.g. out of memory)
                r = {"status": "failed", "error": f"worker exited with code {pr.exitcode} without a result"}
            status[j.jid] = r["status"]
            log({"job": j.jid, "event": "end", "exitcode": pr.exitcode, **r})
            logger.info("job %s -> %s (%.0fs)", j.jid, r["status"], r.get("elapsed_s", float("nan")))
    return status


def _proc_entry(fn: Callable[..., Any], kwargs: dict[str, Any], conn: Any) -> None:
    _init_worker()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    conn.send(_call(fn, kwargs))
    conn.close()
