"""Turn dataset arrays into plain JSON payloads for the Plotly charts."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import tensorly as tl

from ..constant import CROSS
from ..parafac import split_validation


def _num(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def _grid(array: np.ndarray) -> list[list[Optional[float]]]:
    return [[_num(v) for v in row] for row in np.asarray(array)]


def _range(array: np.ndarray) -> dict:
    values = np.asarray(array, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {'min': 0.0, 'max': 0.0}
    return {'min': float(finite.min()), 'max': float(finite.max())}


def _framed(array: np.ndarray) -> dict:
    return {'z': _grid(array), **_range(array)}


def _records(df: pd.DataFrame) -> list[dict]:
    return [
        {str(k): _num(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
         for k, v in row.items()}
        for row in df.to_dict('records')
    ]


def _fri_available(ds) -> bool:
    return bool(
        ds.ex.min() < 250 < ds.ex.max()
        and ds.em.min() < 330
        and ds.em.max() > 380
    )


def history_payload(session) -> dict:
    from .session import MAX_HISTORY
    return {
        'position': session.position,
        'entries': [
            {k: entry[k] for k in ('index', 'action', 'params', 'timestamp')}
            for entry in session.history
        ],
        'max': MAX_HISTORY,
        'canRedo': session.position < len(session.history) - 1,
    }


def state_payload(session) -> dict:
    ds = session.dataset
    base = {
        'imported': ds is not None,
        'position': session.position,
        'historyLength': len(session.history),
        'revision': session.revision,
        'sources': session.sources,
    }
    if ds is None:
        return base
    base.update({
        'nSample': int(ds.nSample),
        'nem': int(ds.nem),
        'nex': int(ds.nex),
        'ex': {'min': float(ds.ex.min()), 'max': float(ds.ex.max()), 'count': int(ds.nex)},
        'em': {'min': float(ds.em.min()), 'max': float(ds.em.max()), 'count': int(ds.nem)},
        'unit': ds.iu,
        'blanked': bool(ds.blanked),
        'ife': bool(ds.ife),
        'hasAbs': session.abs_obj is not None,
        'hasBlank': session.blank is not None,
        'hasDilution': session.dilution is not None,
        'dilution': (
            None if session.dilution is None or len(session.dilution) != ds.nSample
            else [float(i) for i in session.dilution]
        ),
        'facNums': [int(i) for i in ds.fac_nums],
        'selectedF': session.selected_f,
        'coreConsistency': {str(k): float(v) for k, v in ds.core_consistency.items()},
        'explanationRate': {str(k): float(v) for k, v in ds.explanation_rate.items()},
        'hasSplit': bool(ds.split_result),
        'nanCount': int(np.isnan(ds.x).sum()),
        'nanSamples': int(np.isnan(ds.x).any(axis=(1, 2)).sum()),
        'friAvailable': _fri_available(ds),
        'indicesAvailable': bool(ds.nex > 2 and ds.nem > 2),
        'pipeline': list(session.log),
    })
    return base


def samples_payload(session) -> list[dict]:
    ds = session.dataset
    sources = session.sources or {}
    abs_files = sources.get('abs_files') or []
    blank_files = sources.get('blank_files') or []
    dilution = session.dilution
    out = []
    for i in range(ds.nSample):
        out.append({
            'order': int(ds.i[i]) + 1,
            'index': i + 1,
            'file': str(ds.file_list[i]),
            'dilution': None if dilution is None or i >= len(dilution) else float(dilution[i]),
            'abs': abs_files[i] if i < len(abs_files) else None,
            'blank': blank_files[i] if i < len(blank_files) else None,
        })
    return out


def _resolve_state(session, state: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    ds = session.dataset
    if state == 'previous':
        snapshot = session.previous_snapshot()
        if snapshot is not None:
            return snapshot['x'], snapshot['ex'], snapshot['em'], 'Previous step'
    elif state.startswith('index:'):
        index = int(state.split(':', 1)[1])
        from ..project import load_snapshot
        snapshot = load_snapshot(session.history[index]['path'])
        return snapshot['x'], snapshot['ex'], snapshot['em'], f'Step {index}'
    return ds.x, ds.ex, ds.em, 'Current state'


def eem_payload(
        session,
        sample: int,
        state: str = 'current',
        model_f: Optional[int] = None,
        stride: int = 1,
) -> dict:
    ds = session.dataset
    idx = int(sample) - 1
    if idx < 0 or idx >= ds.nSample:
        raise IndexError(f'Sample {sample} is out of range.')
    stride = max(1, int(stride))
    x, ex, em, title = _resolve_state(session, state)
    if model_f is not None:
        model = tl.cp_to_tensor(ds.factors[int(model_f)])
        z = np.asarray(model[idx])
        title = f'Model {model_f}'
        ex, em = ds.ex, ds.em
    else:
        z = np.asarray(x[idx])
    z = z[::stride, ::stride]
    ex = np.asarray(ex)[::stride]
    em = np.asarray(em)[::stride]
    finite = z[np.isfinite(z)]
    return {
        'ex': [float(i) for i in ex],
        'em': [float(i) for i in em],
        'z': _grid(z),
        'zmin': float(finite.min()) if finite.size else 0.0,
        'zmax': float(finite.max()) if finite.size else 0.0,
        'nanCount': int(np.isnan(z).sum()),
        'sample': {'order': int(sample), 'file': str(ds.file_list[idx])},
        'unit': ds.iu,
        'title': title,
    }


def residual_payload(session, f: int, sample: int) -> dict:
    ds = session.dataset
    idx = int(sample) - 1
    if idx < 0 or idx >= ds.nSample:
        raise IndexError(f'Sample {sample} is out of range.')
    model = tl.cp_to_tensor(ds.factors[int(f)])
    observed = np.asarray(ds.x[idx])
    fitted = np.asarray(model[idx])
    residual = observed - fitted
    return {
        'ex': [float(i) for i in ds.ex],
        'em': [float(i) for i in ds.em],
        'sample': _framed(observed),
        'model': _framed(fitted),
        'residual': _framed(residual),
        'sampleInfo': {'order': int(sample), 'file': str(ds.file_list[idx])},
        'title': f'Model {f} for sample {sample}',
    }


def fingers_payload(session, f: int) -> dict:
    ds = session.dataset
    model = ds.factors[int(f)]
    components = []
    for j in range(int(f)):
        z = np.outer(model[1][1][:, j], model[1][2][:, j])
        components.append({'name': f'Component {j + 1}', **_framed(z)})
    peaks = []
    from scipy.signal import find_peaks
    for j in range(int(f)):
        em_peaks = ds.em[find_peaks(model[1][1][:, j])[0]]
        ex_peaks = ds.ex[find_peaks(model[1][2][:, j])[0]]
        peaks.append({
            'component': f'Component {j + 1}',
            'em': [float(i) for i in em_peaks],
            'ex': [float(i) for i in ex_peaks],
        })
    return {
        'ex': [float(i) for i in ds.ex],
        'em': [float(i) for i in ds.em],
        'components': components,
        'peaks': peaks,
    }


def loadings_payload(session, f: int) -> dict:
    ds = session.dataset
    model = ds.factors[int(f)]
    series = []
    for j in range(int(f)):
        series.append({
            'component': f'Component {j + 1}',
            'ex': [float(i) for i in model[1][2][:, j]],
            'em': [float(i) for i in model[1][1][:, j]],
        })
    return {
        'ex': [float(i) for i in ds.ex],
        'em': [float(i) for i in ds.em],
        'series': series,
    }


def fmax_payload(session, f: int) -> dict:
    ds = session.dataset
    model = ds.factors[int(f)]
    fmax = model[1][0] * model[1][1].max(axis=0) * model[1][2].max(axis=0)
    series = [
        {'component': f'Component {j + 1}', 'values': [float(i) for i in fmax[:, j]]}
        for j in range(int(f))
    ]
    return {
        'labels': [str(i) for i in ds.file_list],
        'orders': [int(i) + 1 for i in ds.i],
        'fmax': _grid(fmax),
        'series': series,
    }


def outlier_payload(session, f: int) -> dict:
    ds = session.dataset
    return {
        'f': int(f),
        'samples': [
            {
                'order': int(ds.i[i]) + 1,
                'index': i + 1,
                'file': str(ds.file_list[i]),
                'leverage': float(ds.leverage[int(f)][i]),
                'sse': float(ds.sse[int(f)][i]),
            }
            for i in range(ds.nSample)
        ],
    }


def core_payload(session) -> dict:
    ds = session.dataset
    fs = sorted(int(i) for i in ds.fac_nums)
    return {
        'facNums': fs,
        'coreConsistency': [float(ds.core_consistency[i]) for i in fs],
        'explanationRate': [float(ds.explanation_rate[i]) for i in fs],
        'explanationPercent': [100 * float(ds.explanation_rate[i]) for i in fs],
    }


def sse_payload(session) -> dict:
    ds = session.dataset
    fs = sorted(int(i) for i in ds.fac_nums)
    data = ds.sse_analysis(fs)
    return {
        'facNums': fs,
        'total': [float(data[i]['total']) for i in fs],
        'samples': {
            'labels': [str(i) for i in ds.file_list],
            'series': [{'f': i, 'values': [float(v) for v in data[i]['sample']]} for i in fs],
        },
        'em': {
            'wavelengths': [float(i) for i in ds.em],
            'series': [{'f': i, 'values': [float(v) for v in data[i]['em']]} for i in fs],
        },
        'ex': {
            'wavelengths': [float(i) for i in ds.ex],
            'series': [{'f': i, 'values': [float(v) for v in data[i]['ex']]} for i in fs],
        },
    }


def split_payload(session, f: int) -> dict:
    ds = session.dataset
    f = int(f)
    models = ds.split_result[f]
    letters = 'ABCD'
    comparisons = []
    for c, (first, second) in enumerate(CROSS):
        title = ''.join(letters[i] for i in first) + ' vs ' + ''.join(letters[i] for i in second)
        em_series, ex_series = [], []
        for j in range(f):
            em_series.append({
                'component': f'Component {j + 1}',
                'first': [float(i) for i in models[2 * c][1][1][:, j]],
                'second': [float(i) for i in models[2 * c + 1][1][1][:, j]],
                'overall': [float(i) for i in ds.factors[f][1][1][:, j]],
            })
            ex_series.append({
                'component': f'Component {j + 1}',
                'first': [float(i) for i in models[2 * c][1][2][:, j]],
                'second': [float(i) for i in models[2 * c + 1][1][2][:, j]],
                'overall': [float(i) for i in ds.factors[f][1][2][:, j]],
            })
        comparisons.append({'title': title, 'em': em_series, 'ex': ex_series})
    report = split_validation(ds.factors, ds.split_result, [f])[f]
    return {
        'f': f,
        'em': [float(i) for i in ds.em],
        'ex': [float(i) for i in ds.ex],
        'comparisons': comparisons,
        'tcc': {
            'minTcc': float(report['min_tcc']),
            'validated': bool(report['validated']),
            'emTcc': [float(i) for i in report['em_tcc']],
            'exTcc': [float(i) for i in report['ex_tcc']],
        },
    }


def fri_payload(session, sample: int) -> dict:
    ds = session.dataset
    idx = int(sample) - 1
    if idx < 0 or idx >= ds.nSample:
        raise IndexError(f'Sample {sample} is out of range.')
    z = np.asarray(ds.x[idx])
    return {
        'ex': [float(i) for i in ds.ex],
        'em': [float(i) for i in ds.em],
        'z': _grid(z),
        'zmin': _range(z)['min'],
        'zmax': _range(z)['max'],
        'sample': {'order': int(sample), 'file': str(ds.file_list[idx])},
        'boundaries': {'em': [330.0, 380.0], 'ex': [250.0]},
        'unit': ds.iu,
    }


def indices_payload(session) -> dict:
    df = session.dataset._pick_fluorescence_indices()
    return {
        'columns': [str(c) for c in df.columns],
        'rows': _records(df),
    }


def fri_table_payload(session) -> dict:
    p, phi, mf = session.dataset._fluorescence_regional_integration()
    tables = []
    for name, frame, with_sample in (
            ('percent fluorescence response', pd.DataFrame(p, columns=range(1, 6)), True),
            ('volume', pd.DataFrame(phi, columns=range(1, 6)), True),
            ('multiplication factor', pd.DataFrame(mf, columns=range(1, 6)), False),
    ):
        if with_sample:
            frame.insert(loc=0, column='Sample', value=session.dataset.file_list)
        tables.append({'name': name, 'columns': [str(c) for c in frame.columns], 'rows': _records(frame)})
    return {'tables': tables}


def slope_payload(session) -> dict:
    if session.abs_obj is None:
        raise ValueError('No absorbance data has been imported.')
    df = session.abs_obj.slope_fit()
    df.insert(loc=0, column='Sample', value=[str(i) for i in session.abs_obj.file_list])
    return {
        'columns': [str(c) for c in df.columns],
        'rows': _records(df),
    }
