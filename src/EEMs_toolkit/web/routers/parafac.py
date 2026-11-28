"""PARAFAC / split-half jobs (start, stop, SSE) and chart payloads."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from .. import payloads
from ..jobs import MANAGER, Job
from ..schemas import ParafacJobRequest, SplitJobRequest
from ..session import Session, get_session
from ...parafac import JobCancelled, fit_models, split_fit

router = APIRouter(prefix='/api', tags=['parafac'])


def _check_dataset(session: Session) -> None:
    if session.dataset is None:
        raise HTTPException(status_code=400, detail='No dataset has been imported.')


def _validate_fs(ds, fs: list[int]) -> list[int]:
    if not fs:
        raise HTTPException(status_code=400, detail='At least one component number is required.')
    limit = min(ds.nSample, ds.nem, ds.nex)
    for f in fs:
        if f < 1 or f > limit:
            raise HTTPException(
                status_code=400,
                detail=f'Component number {f} is out of range (1..{limit}).',
            )
    return [int(i) for i in fs]


def _job_state(job: Job) -> dict:
    return job.snapshot()


@router.post('/jobs/parafac')
def start_parafac(request: ParafacJobRequest, session: Session = Depends(get_session)):
    _check_dataset(session)
    fs = _validate_fs(session.dataset, request.f)
    try:
        job = MANAGER.create(session.id, 'parafac')
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    with session.lock:
        nan_count = int(np.isnan(session.dataset.x).sum())
        x = np.nan_to_num(session.dataset.x.copy(), nan=0.0)
        revision = session.revision
    job.total = len(fs) * request.starts

    def progress(done: int, total: int, f: int) -> None:
        job.progress(done, total, f)
        job.add_log(f'A {f}-component model has been fitted.')

    def worker() -> None:
        try:
            result = fit_models(
                x, fs,
                starts=request.starts, tol=request.tol, max_iter=request.max_iter,
                n_processes=request.n_processes,
                progress=progress, stop_event=job.stop_event,
            )
            if job.stop_event.is_set():
                job.state = 'stopped'
                return
            with session.lock:
                if session.revision != revision:
                    job.state = 'stale'
                    job.error = 'The dataset changed while fitting, results were discarded.'
                    return
                ds = session.dataset
                ds.fac_nums = list(fs)
                ds.factors = result.factors
                ds.leverage = result.leverage
                ds.sse = result.sse
                ds.core_consistency = result.core_consistency
                ds.explanation_rate = result.explanation_rate
                ds.split_result = {}
                session.selected_f = fs[0] if session.selected_f not in fs else session.selected_f
                session.revision += 1
                job.revision = session.revision
            job.state = 'done'
        except JobCancelled:
            job.state = 'stopped'
        except Exception as exc:  # noqa: BLE001
            job.state = 'error'
            job.error = str(exc)
        finally:
            job.finished = time.time()

    job.add_log(f'Fitting started: f={fs}, starts={request.starts}.')
    if nan_count:
        job.add_log(f'Note: {nan_count} missing values are treated as 0.')
    threading.Thread(target=worker, name=f'parafac-{job.id}', daemon=True).start()
    return {'job_id': job.id, 'total': job.total}


@router.post('/jobs/split')
def start_split(request: SplitJobRequest, session: Session = Depends(get_session)):
    _check_dataset(session)
    ds = session.dataset
    if not ds.factors:
        raise HTTPException(status_code=400, detail='Run PARAFAC before the split-half analysis.')
    fs = _validate_fs(ds, request.f)
    missing = [f for f in fs if f not in ds.factors]
    if missing:
        raise HTTPException(status_code=400, detail=f'No PARAFAC model for f={missing}.')
    try:
        job = MANAGER.create(session.id, 'split')
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    with session.lock:
        x = np.nan_to_num(ds.x.copy(), nan=0.0)
        factors = dict(ds.factors)
        revision = session.revision
    job.total = len(fs) * 6 * request.starts

    def progress(done: int, total: int, f: int) -> None:
        job.progress(done, total, f)
        job.add_log(f'A {f}-component model has been fitted.')

    def worker() -> None:
        try:
            result = split_fit(
                x, factors, fs,
                starts=request.starts, tol=request.tol, max_iter=request.max_iter,
                random_state=request.random_state, n_processes=request.n_processes,
                progress=progress, stop_event=job.stop_event,
            )
            if job.stop_event.is_set():
                job.state = 'stopped'
                return
            with session.lock:
                if session.revision != revision:
                    job.state = 'stale'
                    job.error = 'The dataset changed while fitting, results were discarded.'
                    return
                session.dataset.split_result = result.split_result
                session.revision += 1
                job.revision = session.revision
            job.state = 'done'
        except JobCancelled:
            job.state = 'stopped'
        except Exception as exc:  # noqa: BLE001
            job.state = 'error'
            job.error = str(exc)
        finally:
            job.finished = time.time()

    job.add_log(f'Split analysis started: f={fs}, starts={request.starts}.')
    threading.Thread(target=worker, name=f'split-{job.id}', daemon=True).start()
    return {'job_id': job.id, 'total': job.total}


@router.get('/jobs/{job_id}')
def get_job(job_id: str):
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found.')
    return _job_state(job)


@router.get('/jobs')
def list_jobs(session: Session = Depends(get_session)):
    active = MANAGER.active(session.id)
    return {'active': None if active is None else _job_state(active)}


@router.post('/jobs/{job_id}/stop')
def stop_job(job_id: str):
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found.')
    MANAGER.stop(job)
    return {'state': job.state, 'stopping': True}


@router.get('/jobs/{job_id}/events')
async def job_events(job_id: str):
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found.')

    async def generator():
        while True:
            yield f"data: {json.dumps(job.snapshot(), ensure_ascii=False)}\n\n"
            if job.state != 'running':
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        generator(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


#  Chart payloads

@router.get('/charts/eem')
def chart_eem(
        sample: int = Query(1, ge=1),
        state: str = 'current',
        model_f: Optional[int] = None,
        stride: int = Query(1, ge=1),
        session: Session = Depends(get_session),
):
    _check_dataset(session)
    try:
        return payloads.eem_payload(session, sample, state=state, model_f=model_f, stride=stride)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@router.get('/charts/residual')
def chart_residual(f: int, sample: int = Query(1, ge=1), session: Session = Depends(get_session)):
    _check_dataset(session)
    if session.dataset.factors.get(f) is None:
        raise HTTPException(status_code=400, detail=f'No PARAFAC model for f={f}.')
    return payloads.residual_payload(session, f, sample)


@router.get('/charts/fingers')
def chart_fingers(f: int, session: Session = Depends(get_session)):
    _check_dataset(session)
    if session.dataset.factors.get(f) is None:
        raise HTTPException(status_code=400, detail=f'No PARAFAC model for f={f}.')
    return payloads.fingers_payload(session, f)


@router.get('/charts/loadings')
def chart_loadings(f: int, session: Session = Depends(get_session)):
    _check_dataset(session)
    if session.dataset.factors.get(f) is None:
        raise HTTPException(status_code=400, detail=f'No PARAFAC model for f={f}.')
    return payloads.loadings_payload(session, f)


@router.get('/charts/fmax')
def chart_fmax(f: int, session: Session = Depends(get_session)):
    _check_dataset(session)
    if session.dataset.factors.get(f) is None:
        raise HTTPException(status_code=400, detail=f'No PARAFAC model for f={f}.')
    return payloads.fmax_payload(session, f)


@router.get('/charts/outlier')
def chart_outlier(f: int, session: Session = Depends(get_session)):
    _check_dataset(session)
    if session.dataset.factors.get(f) is None:
        raise HTTPException(status_code=400, detail=f'No PARAFAC model for f={f}.')
    return payloads.outlier_payload(session, f)


@router.get('/charts/core')
def chart_core(session: Session = Depends(get_session)):
    _check_dataset(session)
    if not session.dataset.fac_nums:
        raise HTTPException(status_code=400, detail='No PARAFAC model has been fitted.')
    return payloads.core_payload(session)


@router.get('/charts/sse')
def chart_sse(session: Session = Depends(get_session)):
    _check_dataset(session)
    if not session.dataset.fac_nums:
        raise HTTPException(status_code=400, detail='No PARAFAC model has been fitted.')
    return payloads.sse_payload(session)


@router.get('/charts/split')
def chart_split(f: int, session: Session = Depends(get_session)):
    _check_dataset(session)
    if session.dataset.split_result.get(f) is None:
        raise HTTPException(status_code=400, detail=f'No split result for f={f}.')
    return payloads.split_payload(session, f)


@router.get('/charts/fri')
def chart_fri(sample: int = Query(1, ge=1), session: Session = Depends(get_session)):
    _check_dataset(session)
    return payloads.fri_payload(session, sample)
