"""Preprocessing endpoints: every call appends a step to the session timeline."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from .. import payloads
from ..schemas import CutScatterRequest, DiluteRequest, RamanRequest, SmoothRequest, SubDatasetRequest
from ..session import HistoryLimitReached, Session, get_session

router = APIRouter(prefix='/api/preprocess', tags=['preprocess'])


def _run(session: Session, action: str, params: dict, fn) -> dict:
    if session.dataset is None:
        raise HTTPException(status_code=400, detail='No dataset has been imported.')
    snapshot = session.dataset.snapshot()
    recorded = False
    try:
        with session.lock:
            fn()
            session.invalidate_models()
            session.record(action, params)
            recorded = True
            session.sync_dilution()
            return {'state': payloads.state_payload(session), 'history': payloads.history_payload(session)}
    except HistoryLimitReached as exc:
        session.dataset.restore(snapshot)
        session.sync_dilution()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        if recorded:
            try:
                session.trim(session.position)
            except Exception:  # noqa: BLE001
                pass
        session.dataset.restore(snapshot)
        session.sync_dilution()
        raise HTTPException(status_code=400, detail=str(exc))


@router.post('/inner_effect_correct')
def inner_effect_correct(session: Session = Depends(get_session)):
    if session.abs_obj is None:
        raise HTTPException(status_code=400, detail='No absorbance data was imported.')
    if session.dataset is not None and session.dataset.ife:
        raise HTTPException(status_code=409, detail='Inner filter effect correction has already been applied.')
    return _run(
        session, 'inner_effect_correct', {},
        lambda: session.dataset.inner_effect_correct(session.abs_obj),
    )


@router.post('/minus_blank')
def minus_blank(session: Session = Depends(get_session)):
    if session.blank is None:
        raise HTTPException(status_code=400, detail='No blank EEM was imported.')
    if session.dataset is not None and session.dataset.blanked:
        raise HTTPException(status_code=409, detail='Blank subtraction has already been applied.')
    return _run(
        session, 'minus_blank', {},
        lambda: session.dataset.minus_the_blank(session.blank),
    )


@router.post('/raman_areal')
def raman_areal(request: RamanRequest, session: Session = Depends(get_session)):
    if session.blank is None:
        raise HTTPException(status_code=400, detail='No blank EEM was imported.')
    if session.dataset is not None and session.dataset.iu == 'RU':
        raise HTTPException(status_code=409, detail='Raman unit conversion has already been applied.')
    params = {'ex': request.ex, 'em_range': list(request.em_range)}
    return _run(
        session, 'raman_areal', params,
        lambda: session.dataset.raman_areal(session.blank, ex=request.ex, em_range=tuple(request.em_range)),
    )


@router.post('/sub_dataset')
def sub_dataset(request: SubDatasetRequest, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(status_code=400, detail='No dataset has been imported.')

    def apply():
        ex_mask = None
        if request.ex_lt is not None or request.ex_gt is not None:
            ex_mask = np.zeros(ds.nex, dtype=bool)
            if request.ex_lt is not None:
                ex_mask |= ds.ex < request.ex_lt
            if request.ex_gt is not None:
                ex_mask |= ds.ex > request.ex_gt
        em_mask = None
        if request.em_lt is not None or request.em_gt is not None:
            em_mask = np.zeros(ds.nem, dtype=bool)
            if request.em_lt is not None:
                em_mask |= ds.em < request.em_lt
            if request.em_gt is not None:
                em_mask |= ds.em > request.em_gt
        ds.sub_dataset(sample_orders=request.sample_orders, ex_orders=ex_mask, em_orders=em_mask)
        if ds.nSample < 1:
            raise ValueError('At least one sample must be kept.')
        if ds.nex < 2 or ds.nem < 2:
            raise ValueError('At least two excitation and two emission wavelengths must be kept.')

    return _run(session, 'sub_dataset', request.model_dump(), apply)


@router.post('/dilute')
def dilute(request: DiluteRequest, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(status_code=400, detail='No dataset has been imported.')
    factors = request.factors
    if factors is None:
        if session.dilution is None:
            raise HTTPException(status_code=400, detail='No dilution factors are available.')
        factors = [float(i) for i in session.dilution]
    values = np.asarray(factors, dtype=np.float64)
    if values.ndim != 1 or values.size != ds.nSample:
        raise HTTPException(
            status_code=400,
            detail=f'Expected {ds.nSample} dilution factors, got {values.size}.',
        )
    if not np.isfinite(values).all() or (values <= 0).any():
        raise HTTPException(status_code=400, detail='Dilution factors must be finite positive numbers.')
    factors = [float(i) for i in values]
    return _run(session, 'dilute', {'factors': factors}, lambda: ds.dilute(factors))


@router.post('/smooth')
def smooth(request: SmoothRequest, session: Session = Depends(get_session)):
    params = {'sigma': request.sigma, 'filter_size': request.filter_size}

    def apply():
        session.dataset.smooth_eem(sigma=request.sigma, filter_size=request.filter_size)

    return _run(session, 'smooth', params, apply)


@router.post('/cut_scatter')
def cut_scatter(request: CutScatterRequest, session: Session = Depends(get_session)):
    params = request.model_dump()

    def apply():
        ds = session.dataset
        ds.cut_ray_scatter(request.ray, request.second_ray)
        ds.cut_ram_scatter(request.ram, request.second_ram)

    return _run(session, 'cut_scatter', params, apply)


@router.post('/interpolate')
def interpolate(session: Session = Depends(get_session)):
    return _run(session, 'interpolate', {}, lambda: session.dataset.miss_value_interpolation())
