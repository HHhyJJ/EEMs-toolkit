from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    id: str
    kind: str
    session_id: str
    state: str = 'running'
    done: int = 0
    total: int = 0
    phase: str = ''
    log: list[str] = field(default_factory=list)
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    revision: int = 0

    def progress(self, done: int, total: int, f: int) -> None:
        self.done = int(done)
        self.total = int(total)
        self.phase = f'{f}-component model fitted'

    def add_log(self, line: str) -> None:
        self.log.append(str(line))
        if len(self.log) > 500:
            del self.log[:len(self.log) - 500]

    def snapshot(self) -> dict:
        elapsed = (self.finished or time.time()) - self.started
        return {
            'id': self.id,
            'kind': self.kind,
            'state': self.state,
            'done': self.done,
            'total': self.total,
            'phase': self.phase,
            'elapsed': round(elapsed, 3),
            'log_tail': self.log[-20:],
            'error': self.error,
            'revision': self.revision,
        }


class JobManager:
    """One long-running job per session at a time."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, kind: str) -> Job:
        with self._lock:
            for job in self._jobs.values():
                if job.session_id == session_id and job.state == 'running':
                    raise RuntimeError('Another job is already running for this session.')
            job = Job(id=uuid.uuid4().hex, kind=kind, session_id=session_id)
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self, session_id: str) -> Optional[Job]:
        with self._lock:
            for job in self._jobs.values():
                if job.session_id == session_id and job.state == 'running':
                    return job
        return None

    def stop(self, job: Job) -> None:
        job.stop_event.set()

    def cleanup(self) -> None:
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.finished is not None and time.time() - job.finished > 3600:
                    self._jobs.pop(job_id, None)


MANAGER = JobManager()
