"""Data import, session state, history timeline and project archives."""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile

from .. import payloads
from ..schemas import HistoryJumpRequest, ProjectSaveRequest
from ..session import Session, get_session
from ... import loader, project

router = APIRouter(prefix='/api', tags=['data'])


@router.get('/session')
def session_info(session: Session = Depends(get_session)):
    """Create or return the browser session before any parallel API request is issued."""
    return {'sid': session.id}


def _reset_dir(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

DEMO_PACKAGE = Path(__file__).resolve().parent.parent / 'demo' / 'eems-demo.zip'


def _fail(exc: Exception, status: int = 400):
    raise HTTPException(status_code=status, detail=str(exc))


def _safe_extract(data: bytes, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.infolist():
            target = (root / member.filename).resolve()
            if not str(target).startswith(str(root.resolve())):
                raise ValueError(f'Unsafe path inside the archive: {member.filename}')
        zf.extractall(root)


def _find_dir(root: Path, name: str) -> Optional[Path]:
    if (root / name).is_dir():
        return root / name
    for path in root.rglob('*'):
        if path.is_dir() and path.name.lower() == name:
            return path
    return None


def _find_log(root: Path) -> Optional[Path]:
    for path in root.rglob('*.xlsx'):
        if path.stem.lower().replace('_', '').replace(' ', '') in ('samplelog', 'log', 'samples'):
            return path
    return None


def _store_uploads(files: list[UploadFile], directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for upload in files:
        name = Path(upload.filename or 'file.xlsx').name
        (directory / name).write_bytes(upload.file.read())
        names.append(name)
    return names


def _import(
        session: Session,
        eem_dir: Path,
        abs_dir: Optional[Path],
        blank_dir: Optional[Path],
        log_path: Optional[Path],
) -> dict:
    if not eem_dir.is_dir():
        raise ValueError(f'EEM directory does not exist: {eem_dir}')
    eem_files: list[str]
    abs_files: Optional[list[str]] = None
    blank_files: Optional[list[str]] = None
    dilution: Optional[np.ndarray] = None
    if log_path is not None:
        log = pd.read_excel(log_path)
        columns = {str(c).strip().lower(): c for c in log.columns}
        if 'eem' not in columns:
            raise ValueError('The sample log must contain an "eem" column.')
        eem_files = [str(i) for i in log[columns['eem']]]
        if 'abs' in columns:
            abs_files = [str(i) for i in log[columns['abs']]]
        if 'blank' in columns:
            blank_files = [str(i) for i in log[columns['blank']]]
        if 'dilution' in columns:
            dilution = log[columns['dilution']].to_numpy(dtype=np.float64)
            if dilution.ndim != 1 or not np.isfinite(dilution).all() or (dilution <= 0).any():
                raise ValueError('The dilution column must contain finite positive numbers.')
    else:
        eem_files = sorted(p.name for p in eem_dir.glob('*.xlsx'))
        if not eem_files:
            raise ValueError(f'No .xlsx EEM file was found in {eem_dir}')

    if abs_dir is not None:
        if abs_files is None:
            candidates = [f'{Path(name).stem}.xlsx' for name in eem_files]
            if all((abs_dir / name).is_file() for name in candidates):
                abs_files = candidates
        else:
            missing = [name for name in abs_files if not (abs_dir / name).is_file()]
            if missing:
                raise ValueError(f'Absorbance file not found: {missing[0]}')
    if blank_dir is not None and blank_files is None:
        blanks = sorted(p.name for p in blank_dir.glob('*.xlsx'))
        if blanks:
            blank_files = [blanks[0]] * len(eem_files)

    ds = loader.read_eems(eem_files, str(eem_dir))
    abs_obj = None
    if abs_dir is not None and abs_files:
        abs_obj = loader.read_abs(abs_files, str(abs_dir))
    blank = None
    if blank_dir is not None and blank_files:
        blank = loader.read_blank(blank_files, str(blank_dir))
    if dilution is None:
        dilution = np.ones(len(eem_files), dtype=np.float64)
    if len(dilution) != ds.nSample:
        raise ValueError(
            f'The dilution column has {len(dilution)} values but {ds.nSample} samples were imported.'
        )
    sources = {
        'eem_dir': str(eem_dir),
        'abs_dir': None if abs_dir is None else str(abs_dir),
        'blank_dir': None if blank_dir is None else str(blank_dir),
        'log': None if log_path is None else str(log_path),
        'eem_files': eem_files,
        'abs_files': abs_files or [],
        'blank_files': blank_files or [],
    }
    session.set_dataset(ds, sources=sources, abs_obj=abs_obj, blank=blank, dilution=dilution)
    return payloads.state_payload(session)


@router.get('/state')
def get_state(session: Session = Depends(get_session)):
    return payloads.state_payload(session)


@router.get('/samples')
def get_samples(session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(status_code=400, detail='No dataset has been imported.')
    return {'samples': payloads.samples_payload(session)}


@router.post('/import')
async def import_files(
        eem_files: list[UploadFile] = File(default=[]),
        abs_files: list[UploadFile] = File(default=[]),
        blank_files: list[UploadFile] = File(default=[]),
        sample_log: Optional[UploadFile] = File(default=None),
        zip_file: Optional[UploadFile] = File(default=None),
        session: Session = Depends(get_session),
):
    try:
        root = Path(session.upload_dir)
        if zip_file is not None and zip_file.filename:
            archive_dir = root / 'archive'
            if archive_dir.exists():
                shutil.rmtree(archive_dir)
            _safe_extract(zip_file.file.read(), archive_dir)
            eem_dir = _find_dir(archive_dir, 'eem')
            if eem_dir is None:
                raise ValueError('The archive must contain an "eem" directory.')
            abs_dir = _find_dir(archive_dir, 'abs')
            blank_dir = _find_dir(archive_dir, 'blank')
            log_path = _find_log(archive_dir)
        else:
            if not eem_files:
                raise ValueError('Please upload the EEM files or a zip archive.')
            eem_dir = root / 'eem'
            _reset_dir(eem_dir)
            _store_uploads(eem_files, eem_dir)
            abs_dir = None
            blank_dir = None
            if abs_files:
                abs_dir = root / 'abs'
                _reset_dir(abs_dir)
                _store_uploads(abs_files, abs_dir)
            if blank_files:
                blank_dir = root / 'blank'
                _reset_dir(blank_dir)
                _store_uploads(blank_files, blank_dir)
            log_path = None
            if sample_log is not None and sample_log.filename:
                log_path = root / Path(sample_log.filename).name
                log_path.write_bytes(sample_log.file.read())
        return _import(session, eem_dir, abs_dir, blank_dir, log_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@router.get('/import/demo-package')
def demo_package():
    """Download the demo archive (EEM / abs / blank / SampleLog) used in the documentation."""
    if not DEMO_PACKAGE.is_file():
        raise HTTPException(status_code=404, detail='The demo package is not available in this installation.')
    return Response(
        content=DEMO_PACKAGE.read_bytes(),
        media_type='application/zip',
        headers={'Content-Disposition': 'attachment; filename="eems-demo.zip"'},
    )


@router.get('/history')
def get_history(session: Session = Depends(get_session)):
    return payloads.history_payload(session)


@router.post('/history/jump')
def jump_history(request: HistoryJumpRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(status_code=400, detail='No dataset has been imported.')
    try:
        session.jump(request.index)
        return {'state': payloads.state_payload(session), 'history': get_history(session)}
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@router.post('/history/trim')
def trim_history(request: HistoryJumpRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(status_code=400, detail='No dataset has been imported.')
    try:
        session.trim(request.index)
        return {'state': payloads.state_payload(session), 'history': get_history(session)}
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@router.post('/project/save')
def save_project(request: ProjectSaveRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(status_code=400, detail='No dataset has been imported.')
    selected_f = request.f if request.f is not None else session.selected_f
    buf = io.BytesIO()
    try:
        project.dump_project(
            buf,
            session.dataset,
            history=session.history,
            log=session.log,
            selected_f=selected_f,
            sources=session.sources,
            dilution=session.dilution,
            abs_obj=session.abs_obj,
            blank=session.blank,
            pipeline=session.log,
            history_index=session.position,
            include_history=request.include_history,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    headers = {'Content-Disposition': 'attachment; filename="eems-project.eemproj"'}
    return Response(content=buf.getvalue(), media_type='application/zip', headers=headers)


@router.post('/project/load')
async def load_project(file: UploadFile = File(...), session: Session = Depends(get_session)):
    try:
        data = file.file.read()
        staging = Path(session.dir) / 'project_history'
        if staging.exists():
            shutil.rmtree(staging)
        loaded = project.load_project(io.BytesIO(data), history_dir=str(staging))
        session.replace_from_project(loaded)
        shutil.rmtree(staging, ignore_errors=True)
        return {'state': payloads.state_payload(session), 'history': get_history(session)}
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
