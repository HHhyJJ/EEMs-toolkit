from __future__ import annotations

import datetime
import os
import shutil
import tempfile
import threading
import time
import uuid
from typing import Any, Optional

import numpy as np
from fastapi import Request

from ..project import apply_snapshot, load_snapshot, save_snapshot, take_snapshot

MAX_HISTORY = 50
SESSION_TTL = 4 * 3600
MAX_SESSIONS = 20


def _now() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class HistoryLimitReached(Exception):
    """Raised when the timeline already holds MAX_HISTORY steps."""


class Session:
    """State of one browser session: dataset, timeline and imported sources."""

    def __init__(self, sid: Optional[str] = None):
        self.id = sid or uuid.uuid4().hex
        self.dataset = None
        self.abs_obj = None
        self.blank = None
        self.dilution = None
        self.dilution_map: dict[str, float] = {}
        self.sources: dict[str, Any] = {}
        self.history: list[dict] = []
        self.log: list[dict] = []
        self.position = -1
        self.selected_f: Optional[int] = None
        self.revision = 0
        self.lock = threading.RLock()
        self.dir = tempfile.mkdtemp(prefix='eems_web_')
        self.created = time.time()
        self.touched = time.time()

    #  Lifecycle
    def touch(self) -> None:
        self.touched = time.time()

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    @property
    def upload_dir(self) -> str:
        path = os.path.join(self.dir, 'uploads')
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def history_dir(self) -> str:
        path = os.path.join(self.dir, 'history')
        os.makedirs(path, exist_ok=True)
        return path

    #  Dataset
    def set_dataset(self, dataset, *, sources=None, abs_obj=None, blank=None, dilution=None) -> None:
        with self.lock:
            self.dataset = dataset
            self.sources = sources or {}
            self.abs_obj = abs_obj
            self.blank = blank
            self.dilution = dilution
            self.dilution_map = self._dilution_map(dataset, dilution)
            self.selected_f = None
            self.history = []
            self.log = []
            self.position = -1
            self.revision = 0
            for name in os.listdir(self.history_dir):
                os.remove(os.path.join(self.history_dir, name))
            self._append('import', {})

    @staticmethod
    def _dilution_map(dataset, dilution) -> dict[str, float]:
        if dataset is None or dilution is None:
            return {}
        values = np.asarray(dilution, dtype=np.float64)
        if values.ndim != 1 or values.size != dataset.nSample:
            return {}
        return {str(name): float(value) for name, value in zip(dataset.file_list, values)}

    def sync_dilution(self) -> None:
        """Keep dilution factors aligned with the current sample list after cropping."""
        if self.dataset is None or not self.dilution_map:
            return
        values = [self.dilution_map.get(str(name)) for name in self.dataset.file_list]
        if any(value is None for value in values):
            self.dilution = None
            return
        self.dilution = np.asarray(values, dtype=np.float64)

    def invalidate_models(self) -> None:
        ds = self.dataset
        if ds is None:
            return
        ds.fac_nums = []
        ds.factors = {}
        ds.leverage = {}
        ds.sse = {}
        ds.core_consistency = {}
        ds.explanation_rate = {}
        ds.split_result = {}
        self.selected_f = None

    def record(self, action: str, params: Optional[dict] = None) -> None:
        """Append the current dataset state to the timeline after a successful step."""
        with self.lock:
            if self.dataset is None:
                raise RuntimeError('No dataset has been imported.')
            if len(self.history) >= MAX_HISTORY + 1:
                raise HistoryLimitReached(
                    f'The timeline already holds {MAX_HISTORY} steps, save the project before continuing.'
                )
            self._append(action, params or {})
            self.revision += 1

    def _append(self, action: str, params: dict) -> None:
        index = self.position + 1
        for entry in self.history[index:]:
            try:
                os.remove(entry['path'])
            except OSError:
                pass
        self.history = self.history[:index]
        path = os.path.join(self.history_dir, f'{index:03d}.npz')
        save_snapshot(path, take_snapshot(self.dataset))
        entry = {
            'index': index,
            'action': action,
            'params': params,
            'timestamp': _now(),
            'path': path,
        }
        self.history.append(entry)
        self.position = index
        self.log.append({'action': action, 'params': params, 'timestamp': entry['timestamp']})

    def jump(self, index: int) -> None:
        with self.lock:
            if index < 0 or index >= len(self.history):
                raise IndexError(f'History index {index} is out of range.')
            entry = self.history[index]
            apply_snapshot(self.dataset, load_snapshot(entry['path']))
            self.position = index
            self.invalidate_models()
            self.sync_dilution()
            self.revision += 1

    def trim(self, index: int) -> None:
        """Delete the step at ``index`` and every later step."""
        with self.lock:
            if index <= 0:
                raise ValueError('The import step cannot be deleted.')
            if index >= len(self.history):
                return
            for entry in self.history[index:]:
                try:
                    os.remove(entry['path'])
                except OSError:
                    pass
            self.history = self.history[:index]
            self.log = [
                {'action': entry['action'], 'params': entry['params'], 'timestamp': entry['timestamp']}
                for entry in self.history
            ]
            if self.position >= index:
                self.position = index - 1
                apply_snapshot(self.dataset, load_snapshot(self.history[self.position]['path']))
                self.invalidate_models()
            self.sync_dilution()
            self.revision += 1

    def previous_snapshot(self) -> Optional[dict]:
        """Snapshot of the step before the current position, for before/after previews."""
        if self.position <= 0:
            return None
        entry = self.history[self.position - 1]
        return load_snapshot(entry['path'])

    def replace_from_project(self, loaded) -> None:
        """Adopt a LoadedProject without copying its data again."""
        with self.lock:
            self.dataset = loaded.dataset
            self.abs_obj = loaded.abs_obj
            self.blank = loaded.blank
            self.dilution = loaded.dilution
            self.dilution_map = self._dilution_map(loaded.dataset, loaded.dilution)
            self.sources = loaded.sources or {}
            for name in os.listdir(self.history_dir):
                os.remove(os.path.join(self.history_dir, name))
            self.history = []
            for entry in loaded.history:
                path = entry.get('path')
                if path is None:
                    continue
                target = os.path.join(self.history_dir, f"{int(entry['index']):03d}.npz")
                shutil.copyfile(path, target)
                entry = dict(entry)
                entry['path'] = target
                self.history.append(entry)
            if not self.history:
                self.position = -1
                self._append('import', {})
            else:
                self.history.sort(key=lambda e: int(e['index']))
                self.position = min(int(loaded.history_index), len(self.history) - 1)
                apply_snapshot(self.dataset, load_snapshot(self.history[self.position]['path']))
            self.log = list(loaded.log)
            self.selected_f = loaded.selected_f
            self.revision += 1


class SessionStore:
    """Bounded, TTL-based session registry."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, sid: Optional[str]) -> Optional[Session]:
        if not sid:
            return None
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                return None
            if time.time() - session.touched > SESSION_TTL:
                self._sessions.pop(sid, None)
                session.close()
                return None
            session.touch()
            return session

    def get_or_create(self, sid: Optional[str] = None) -> Session:
        session = self.get(sid)
        if session is not None:
            return session
        self.cleanup()
        with self._lock:
            if sid and sid in self._sessions:
                return self._sessions[sid]
            session = Session(sid)
            self._sessions[session.id] = session
            return session

    def cleanup(self) -> None:
        now = time.time()
        with self._lock:
            for sid, session in list(self._sessions.items()):
                if now - session.touched > SESSION_TTL:
                    self._sessions.pop(sid, None)
                    session.close()
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = sorted(self._sessions.values(), key=lambda s: s.touched)
                for session in oldest[:len(self._sessions) - MAX_SESSIONS + 1]:
                    self._sessions.pop(session.id, None)
                    session.close()

    def close_all(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()


STORE = SessionStore()


def get_session(request: Request) -> Session:
    """FastAPI dependency returning the session bound by the middleware."""
    session = getattr(request.state, 'session', None)
    if session is None:
        session = STORE.get_or_create(request.cookies.get('eems_sid'))
    return session
