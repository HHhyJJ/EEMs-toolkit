"""Snapshot helpers and the ``.eemproj`` project archive format."""

from __future__ import annotations

import datetime
import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import tensorly as tl
from tensorly.cp_tensor import CPTensor

FORMAT_VERSION = 1

_SNAPSHOT_ARRAYS = ('x', 'ex', 'em', 'file_list', 'i')
_SNAPSHOT_SCALARS = ('iu', 'blanked', 'ife')


def take_snapshot(dataset) -> dict:
    """Deep copy the mutable state of a dataset."""
    out: dict[str, Any] = {name: np.array(getattr(dataset, name), copy=True) for name in _SNAPSHOT_ARRAYS}
    for name in _SNAPSHOT_SCALARS:
        out[name] = getattr(dataset, name)
    return out


def apply_snapshot(dataset, snapshot: dict) -> None:
    """Restore a dataset in place from :func:`take_snapshot` output."""
    for name in _SNAPSHOT_ARRAYS:
        setattr(dataset, name, np.array(snapshot[name], copy=True))
    dataset.nSample = dataset.x.shape[0]
    dataset.nem = dataset.em.shape[0]
    dataset.nex = dataset.ex.shape[0]
    for name in _SNAPSHOT_SCALARS:
        setattr(dataset, name, snapshot[name])


def save_snapshot(fileobj, snapshot: dict) -> None:
    """Save a snapshot into an npz archive (path or binary file object)."""
    np.savez_compressed(fileobj, **snapshot)


def load_snapshot(fileobj) -> dict:
    """Read a snapshot written by :func:`save_snapshot`."""
    with np.load(fileobj, allow_pickle=False) as data:
        out: dict[str, Any] = {name: data[name] for name in _SNAPSHOT_ARRAYS}
        out['iu'] = str(data['iu'])
        out['blanked'] = bool(data['blanked'])
        out['ife'] = bool(data['ife'])
    return out


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_npy(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, np.asarray(array), allow_pickle=False)
    zf.writestr(name, buf.getvalue())


def _read_npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    data = np.load(io.BytesIO(zf.read(name)), allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            return data['arr_0']
    return np.asarray(data)


@dataclass
class LoadedProject:
    """Everything recovered from a ``.eemproj`` archive."""

    dataset: Any
    history: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    selected_f: Optional[int] = None
    sources: dict = field(default_factory=dict)
    dilution: Optional[np.ndarray] = None
    abs_obj: Any = None
    blank: Optional[np.ndarray] = None
    pipeline: list[dict] = field(default_factory=list)
    history_index: int = 0


def dump_project(
        fileobj,
        dataset,
        *,
        history: Optional[list[dict]] = None,
        log: Optional[list[dict]] = None,
        selected_f: Optional[int] = None,
        sources: Optional[dict] = None,
        dilution: Optional[np.ndarray] = None,
        abs_obj=None,
        blank: Optional[np.ndarray] = None,
        pipeline: Optional[list[dict]] = None,
        history_index: int = 0,
        include_history: bool = True,
) -> None:
    """
    Write a ``.eemproj`` archive.

    :param fileobj: A path or a binary file object.
    :param history: Step metadata; each entry may hold ``path`` pointing to a snapshot npz.
    :param log: Pipeline log entries shown in the UI.
    """
    history = list(history or [])
    meta = {
        'format_version': FORMAT_VERSION,
        'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'nSample': int(dataset.nSample),
        'nem': int(dataset.nem),
        'nex': int(dataset.nex),
        'ex': np.asarray(dataset.ex).tolist(),
        'em': np.asarray(dataset.em).tolist(),
        'file_list': [str(i) for i in dataset.file_list],
        'i': np.asarray(dataset.i).tolist(),
        'iu': dataset.iu,
        'blanked': bool(dataset.blanked),
        'ife': bool(dataset.ife),
        'fac_nums': [int(i) for i in dataset.fac_nums],
        'selected_f': selected_f,
        'core_consistency': _to_jsonable(dataset.core_consistency),
        'explanation_rate': _to_jsonable(dataset.explanation_rate),
        'sources': _to_jsonable(sources or {}),
        'dilution': None if dilution is None else np.asarray(dilution).tolist(),
        'pipeline': _to_jsonable(pipeline or []),
        'history_index': int(history_index),
        'history': [
            {k: v for k, v in _to_jsonable(entry).items() if k != 'path'}
            for entry in history
        ],
        'has_history': bool(include_history and any(entry.get('path') for entry in history)),
        'log': _to_jsonable(log or []),
    }
    with zipfile.ZipFile(fileobj, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('meta.json', json.dumps(meta, ensure_ascii=False, indent=1))
        _write_npy(zf, 'x.npy', dataset.x)
        for f, model in dataset.factors.items():
            _write_npy(zf, f'factors/f{f}/weights.npy', model[0])
            _write_npy(zf, f'factors/f{f}/sample.npy', model[1][0])
            _write_npy(zf, f'factors/f{f}/em.npy', model[1][1])
            _write_npy(zf, f'factors/f{f}/ex.npy', model[1][2])
            if f in dataset.leverage:
                _write_npy(zf, f'factors/f{f}/leverage.npy', dataset.leverage[f])
            if f in dataset.sse:
                _write_npy(zf, f'factors/f{f}/sse.npy', dataset.sse[f])
        for f, models in dataset.split_result.items():
            for j, model in enumerate(models):
                prefix = f'split/f{f}/m{j + 1:02d}'
                _write_npy(zf, f'{prefix}/weights.npy', model[0])
                _write_npy(zf, f'{prefix}/sample.npy', model[1][0])
                _write_npy(zf, f'{prefix}/em.npy', model[1][1])
                _write_npy(zf, f'{prefix}/ex.npy', model[1][2])
        if abs_obj is not None:
            buf = io.BytesIO()
            np.savez(
                buf,
                wavelength=np.asarray(abs_obj.wavelength),
                absorbance=np.asarray(abs_obj.absorbance),
                file_list=np.asarray([str(i) for i in abs_obj.file_list]),
            )
            zf.writestr('abs.npz', buf.getvalue())
        if blank is not None:
            _write_npy(zf, 'blank.npy', np.asarray(blank))
        if include_history:
            for entry in history:
                path = entry.get('path')
                if path:
                    with open(path, 'rb') as fp:
                        zf.writestr(f'history/{int(entry["index"]):03d}.npz', fp.read())


def load_project(fileobj, *, history_dir: Optional[str] = None) -> LoadedProject:
    """
    Read a ``.eemproj`` archive.

    :param fileobj: A path or a binary file object.
    :param history_dir: Optional directory receiving the extracted history snapshots.
    """
    from .eem import EEMsDataset  # local import avoids a circular import at module load

    with zipfile.ZipFile(fileobj, 'r') as zf:
        names = set(zf.namelist())
        if 'meta.json' not in names:
            raise ValueError('Invalid project file: meta.json is missing.')
        meta = json.loads(zf.read('meta.json').decode('utf-8'))
        if int(meta.get('format_version', 0)) != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported project format version: {meta.get('format_version')} "
                f"(expected {FORMAT_VERSION})."
            )
        x = _read_npy(zf, 'x.npy')
        expected = (int(meta['nSample']), int(meta['nem']), int(meta['nex']))
        if tuple(x.shape) != expected:
            raise ValueError(f'x.npy shape {tuple(x.shape)} does not match meta {expected}.')
        ex = np.asarray(meta['ex'], dtype=np.float64)
        em = np.asarray(meta['em'], dtype=np.float64)
        dataset = EEMsDataset(x.copy(), ex, em, meta['file_list'])
        # EEMsDataset zeroes em < ex cells on construction; the archive already holds the
        # final, post-interpolation values, so restore them untouched.
        dataset.x = x
        dataset.i = np.asarray(meta['i'], dtype=np.int64)
        dataset.iu = meta.get('iu', 'AU')
        dataset.blanked = bool(meta.get('blanked', False))
        dataset.ife = bool(meta.get('ife', False))
        dataset.fac_nums = [int(i) for i in meta.get('fac_nums', [])]
        dataset.core_consistency = {int(k): float(v) for k, v in meta.get('core_consistency', {}).items()}
        dataset.explanation_rate = {int(k): float(v) for k, v in meta.get('explanation_rate', {}).items()}
        for f in dataset.fac_nums:
            prefix = f'factors/f{f}'
            if f'{prefix}/weights.npy' not in names:
                continue
            weights = _read_npy(zf, f'{prefix}/weights.npy')
            sample_l = _read_npy(zf, f'{prefix}/sample.npy')
            em_l = _read_npy(zf, f'{prefix}/em.npy')
            ex_l = _read_npy(zf, f'{prefix}/ex.npy')
            dataset.factors[f] = CPTensor((weights, [sample_l, em_l, ex_l]))
            if f'{prefix}/leverage.npy' in names:
                dataset.leverage[f] = _read_npy(zf, f'{prefix}/leverage.npy')
            if f'{prefix}/sse.npy' in names:
                dataset.sse[f] = _read_npy(zf, f'{prefix}/sse.npy')
        for f in sorted({int(i) for i in meta.get('fac_nums', [])}):
            prefix = f'split/f{f}'
            models: list[CPTensor] = []
            for j in range(1, 100):
                sub = f'{prefix}/m{j:02d}'
                if f'{sub}/weights.npy' not in names:
                    break
                weights = _read_npy(zf, f'{sub}/weights.npy')
                sample_l = _read_npy(zf, f'{sub}/sample.npy')
                em_l = _read_npy(zf, f'{sub}/em.npy')
                ex_l = _read_npy(zf, f'{sub}/ex.npy')
                models.append(CPTensor((weights, [sample_l, em_l, ex_l])))
            if models:
                dataset.split_result[f] = models
        abs_obj = None
        if 'abs.npz' in names:
            from .absorbance import Abs
            with np.load(io.BytesIO(zf.read('abs.npz')), allow_pickle=False) as data:
                abs_obj = Abs(data['wavelength'], data['absorbance'], [str(i) for i in data['file_list']])
        blank = _read_npy(zf, 'blank.npy') if 'blank.npy' in names else None
        history = list(meta.get('history', []))
        if history_dir is not None:
            import os
            os.makedirs(history_dir, exist_ok=True)
            for entry in history:
                index = int(entry['index'])
                member = f'history/{index:03d}.npz'
                if member in names:
                    path = os.path.join(history_dir, f'{index:03d}.npz')
                    with open(path, 'wb') as fp:
                        fp.write(zf.read(member))
                    entry['path'] = path
    return LoadedProject(
        dataset=dataset,
        history=history,
        log=list(meta.get('log', [])),
        selected_f=None if meta.get('selected_f') is None else int(meta['selected_f']),
        sources=dict(meta.get('sources', {})),
        dilution=None if meta.get('dilution') is None else np.asarray(meta['dilution'], dtype=np.float64),
        abs_obj=abs_obj,
        blank=blank,
        pipeline=list(meta.get('pipeline', [])),
        history_index=int(meta.get('history_index', 0)),
    )


def rebuild_cp_tensor(weights: np.ndarray, sample: np.ndarray, em: np.ndarray, ex: np.ndarray) -> CPTensor:
    """Build a CPTensor with the factor order used by this package: sample, em, ex."""
    return CPTensor((tl.tensor(weights), [sample, em, ex]))
