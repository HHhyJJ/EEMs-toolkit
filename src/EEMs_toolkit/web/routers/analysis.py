"""Analysis panels (indices, FRI, slopes), outlier removal and the export centre."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from .. import payloads
from ..schemas import ExcludeSamplesRequest
from ..session import HistoryLimitReached, Session, get_session
from ... import export as export_builder

router = APIRouter(prefix='/api', tags=['analysis'])

XLSX_MEDIA = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def _dataset(session: Session):
    if session.dataset is None:
        raise HTTPException(status_code=400, detail='No dataset has been imported.')
    return session.dataset


def _download(content: bytes, filename: str, media_type: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@router.get('/indices')
def get_indices(session: Session = Depends(get_session)):
    _dataset(session)
    try:
        return payloads.indices_payload(session)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@router.get('/fri')
def get_fri(session: Session = Depends(get_session)):
    ds = _dataset(session)
    if not payloads._fri_available(ds):
        raise HTTPException(
            status_code=400,
            detail='FRI requires Ex < 250 nm and Em spanning 330-380 nm; the current ranges do not cover them.',
        )
    try:
        return payloads.fri_table_payload(session)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@router.get('/slope')
def get_slope(session: Session = Depends(get_session)):
    if session.abs_obj is None:
        raise HTTPException(status_code=400, detail='No absorbance data was imported.')
    try:
        return payloads.slope_payload(session)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@router.get('/outliers')
def get_outliers(f: int, session: Session = Depends(get_session)):
    ds = _dataset(session)
    if ds.factors.get(f) is None:
        raise HTTPException(status_code=400, detail=f'No PARAFAC model for f={f}.')
    return payloads.outlier_payload(session, f)


@router.post('/samples/exclude')
def exclude_samples(request: ExcludeSamplesRequest, session: Session = Depends(get_session)):
    ds = _dataset(session)
    if not request.orders:
        raise HTTPException(status_code=400, detail='No sample was selected.')
    snapshot = ds.snapshot()
    recorded = False
    try:
        with session.lock:
            ds.sub_dataset(sample_orders=request.orders)
            if ds.nSample < 1:
                raise ValueError('At least one sample must be kept.')
            session.invalidate_models()
            session.record('exclude_samples', {'orders': request.orders})
            recorded = True
            session.sync_dilution()
            return {'state': payloads.state_payload(session), 'history': payloads.history_payload(session)}
    except HistoryLimitReached as exc:
        ds.restore(snapshot)
        session.sync_dilution()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        if recorded:
            try:
                session.trim(session.position)
            except Exception:  # noqa: BLE001
                pass
        ds.restore(snapshot)
        session.sync_dilution()
        raise HTTPException(status_code=400, detail=str(exc))


@router.get('/export/openfluor')
def export_openfluor(
        f: int,
        name: str = '',
        creator: str = '',
        email: str = '',
        doi: str = '',
        reference: str = '',
        fluorometer: str = '',
        sources: str = '',
        ecozones: str = '',
        method: str = '',
        preprocess: str = '',
        description: str = '',
        session: Session = Depends(get_session),
):
    ds = _dataset(session)
    if ds.factors.get(f) is None:
        raise HTTPException(status_code=400, detail=f'No PARAFAC model for f={f}.')
    info = {
        'name': name, 'creator': creator, 'email': email, 'doi': doi,
        'reference': reference, 'fluorometer': fluorometer, 'sources': sources,
        'ecozones': ecozones, 'method': method, 'preprocess': preprocess,
        'description': description,
    }
    return _download(export_builder.open_fluor_text(ds, f, **info), 'model.txt', 'text/plain; charset=utf-8')


@router.get('/export/parafac')
def export_parafac(f: int, session: Session = Depends(get_session)):
    ds = _dataset(session)
    if ds.factors.get(f) is None:
        raise HTTPException(status_code=400, detail=f'No PARAFAC model for f={f}.')
    return _download(export_builder.parafac_xlsx_bytes(ds, f), 'parafac-result.xlsx', XLSX_MEDIA)


@router.get('/export/eems')
def export_eems(session: Session = Depends(get_session)):
    ds = _dataset(session)
    return _download(export_builder.eems_zip_bytes(ds), 'eems.zip', 'application/zip')


@router.get('/export/indices')
def export_indices(session: Session = Depends(get_session)):
    ds = _dataset(session)
    try:
        content = export_builder.indices_xlsx_bytes(ds)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return _download(content, 'fluorescence_indices.xlsx', XLSX_MEDIA)


@router.get('/export/fri')
def export_fri(session: Session = Depends(get_session)):
    ds = _dataset(session)
    try:
        content = export_builder.fri_xlsx_bytes(ds)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return _download(content, 'FRI.xlsx', XLSX_MEDIA)


@router.get('/export/slope')
def export_slope(session: Session = Depends(get_session)):
    if session.abs_obj is None:
        raise HTTPException(status_code=400, detail='No absorbance data was imported.')
    return _download(export_builder.slope_xlsx_bytes(session.abs_obj), 'slope_fit_result.xlsx', XLSX_MEDIA)


@router.get('/export/all')
def export_all(
        f: int | None = Query(default=None),
        include: str | None = Query(default=None),
        session: Session = Depends(get_session),
):
    ds = _dataset(session)
    selected = None if include is None else [i.strip() for i in include.split(',') if i.strip()]
    content = export_builder.all_zip_bytes(ds, abs_obj=session.abs_obj, f=f, include=selected)
    return _download(content, 'eems-export.zip', 'application/zip')
