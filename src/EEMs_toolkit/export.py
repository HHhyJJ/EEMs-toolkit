"""Builders that turn dataset results into bytes, shared by the file writers and the web API."""

from __future__ import annotations

import datetime
import io
import zipfile
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def open_fluor_text(dataset, f: int, **kwargs) -> bytes:
    """Build the OpenFluor text file content."""
    ex_str = ''
    for i in range(dataset.nex):
        ex_str += f'EX\t{dataset.ex[i]}\t'
        s = '\t'.join(f'{dataset.factors[f][1][2][i, j]}' for j in range(f))
        ex_str = ex_str + s + '\n'
    em_str = ''
    for i in range(dataset.nem):
        em_str += f'EM\t{dataset.em[i]}\t'
        s = '\t'.join(f'{dataset.factors[f][1][1][i, j]}' for j in range(f))
        em_str = em_str + s + '\n'
    em_str = em_str[:-1]
    name = kwargs.get('name', '')
    creator = kwargs.get('creator', '')
    email = kwargs.get('email', '')
    doi = kwargs.get('doi', '')
    reference = kwargs.get('reference', '')
    fluorometer = kwargs.get('fluorometer', '')
    sources = kwargs.get('sources', '')
    ecozones = kwargs.get('ecozones', '')
    method = kwargs.get('method', '')
    preprocess = kwargs.get('preprocess', '')
    description = kwargs.get('description', '')
    text = (f'#\n# Fluorescence Model\n#\n'
            f'name\t{name}\ncreator\t{creator}\nemail\t{email}\ndoi\t{doi}\n'
            f'reference\t{reference}\nunit\t{dataset.iu}\ntoolbox\tEEM toolkit for Python\n'
            f'date\t{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'fluorometer\t{fluorometer}\nnSample\t{dataset.nSample}\nconstraints\tnon-negativity\n'
            f'validation\tSplit-half analysis (S4C6T3)\nmethods\t{method}\n'
            f'preprocess\t{preprocess}\nsources\t{sources}\necozones\t{ecozones}\n'
            f'description\t{description}\n'
            f'#\n# Excitation/Emission (Ex, Em), wavelength (nm), component[n] (intensity)\n#\n'
            f'{ex_str}{em_str}')
    return text.encode('utf-8')


def parafac_xlsx_bytes(dataset, f: int) -> bytes:
    """Build the PARAFAC result workbook (FMax, loadings and component maps)."""
    fmax = pd.DataFrame(
        dataset.factors[f][1][0] * dataset.factors[f][1][1].max(axis=0) * dataset.factors[f][1][2].max(axis=0),
        columns=[f'FMax{i + 1}' for i in range(f)])
    fmax.insert(loc=0, column='EEM', value=dataset.file_list)
    em_loadings = pd.DataFrame(dataset.factors[f][1][1], columns=[f'Em{i + 1}' for i in range(f)])
    em_loadings.insert(loc=0, column='Em', value=dataset.em)
    ex_loadings = pd.DataFrame(dataset.factors[f][1][2], columns=[f'Ex{i + 1}' for i in range(f)])
    ex_loadings.insert(loc=0, column='Ex', value=dataset.ex)
    out_model = []
    for i in range(f):
        out = np.full((dataset.nem + 1, dataset.nex + 1), np.nan)
        j = np.outer(
            dataset.factors[f][1][1][:, i],
            dataset.factors[f][1][2][:, i]
        )
        out[1:, 1:] = j
        out[0, 1:] = dataset.ex
        out[1:, 0] = dataset.em
        out_model.append(pd.DataFrame(out))
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        fmax.to_excel(writer, sheet_name='FMax', index=False)
        em_loadings.to_excel(writer, sheet_name='Em loadings', index=False)
        ex_loadings.to_excel(writer, sheet_name='Ex loadings', index=False)
        for j in range(len(out_model)):
            out_model[j].to_excel(writer, sheet_name=f'C{j + 1}', header=False, index=False)
    return buf.getvalue()


def eem_xlsx_bytes(dataset, index: int) -> bytes:
    """Build one EEM workbook with Ex on the first row and Em on the first column."""
    out = np.full((dataset.nem + 1, dataset.nex + 1), np.nan)
    out[1:, 1:] = dataset.x[index]
    out[0, 1:] = dataset.ex
    out[1:, 0] = dataset.em
    buf = io.BytesIO()
    pd.DataFrame(out).to_excel(buf, header=False, index=False)
    return buf.getvalue()


def eems_zip_bytes(dataset) -> bytes:
    """Build a zip archive of all EEM workbooks."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i in range(dataset.nSample):
            zf.writestr(f'{dataset.file_list[i]}.xlsx', eem_xlsx_bytes(dataset, i))
    return buf.getvalue()


def fri_xlsx_bytes(dataset) -> bytes:
    """Build the FRI workbook. Raises ValueError when the wavelength ranges are not covered."""
    p, phi, mf = dataset._fluorescence_regional_integration()
    cols = np.arange(1, 6)
    pfr = pd.DataFrame(p, columns=cols)
    pfr.insert(loc=0, column='Sample', value=dataset.file_list)
    volume = pd.DataFrame(phi, columns=cols)
    volume.insert(loc=0, column='Sample', value=dataset.file_list)
    mf = pd.DataFrame(mf, columns=cols)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        pfr.to_excel(writer, sheet_name='percent fluorescence response', index=False)
        volume.to_excel(writer, sheet_name='volume', index=False)
        mf.to_excel(writer, sheet_name='multiplication factor', index=False)
    return buf.getvalue()


def indices_xlsx_bytes(dataset) -> bytes:
    """Build the fluorescence indices workbook."""
    buf = io.BytesIO()
    dataset._pick_fluorescence_indices().to_excel(buf, index=False)
    return buf.getvalue()


def slope_xlsx_bytes(abs_obj) -> bytes:
    """Build the absorbance slope workbook."""
    buf = io.BytesIO()
    abs_obj.slope_fit().to_excel(buf, index=False)
    return buf.getvalue()


DEFAULT_EXPORTS = ('openfluor', 'parafac', 'eems', 'indices', 'fri', 'slope')


def all_zip_bytes(
        dataset,
        abs_obj=None,
        f: Optional[int] = None,
        include: Optional[Iterable[str]] = None,
        **kwargs,
) -> bytes:
    """
    Build a zip archive with every requested export.

    Unavailable items are listed in ``skipped.txt`` instead of failing the whole archive.
    """
    wanted = list(DEFAULT_EXPORTS if include is None else include)
    skipped: list[str] = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if 'openfluor' in wanted:
            if f is None:
                skipped.append('openfluor.txt (no component number selected)')
            else:
                zf.writestr('openfluor.txt', open_fluor_text(dataset, f, **kwargs))
        if 'parafac' in wanted:
            if f is None:
                skipped.append('parafac-result.xlsx (no component number selected)')
            else:
                zf.writestr('parafac-result.xlsx', parafac_xlsx_bytes(dataset, f))
        if 'eems' in wanted:
            for i in range(dataset.nSample):
                zf.writestr(f'{dataset.file_list[i]}.xlsx', eem_xlsx_bytes(dataset, i))
        if 'indices' in wanted:
            try:
                zf.writestr('fluorescence_indices.xlsx', indices_xlsx_bytes(dataset))
            except Exception as exc:  # noqa: BLE001 - an unavailable panel should not break the archive
                skipped.append(f'fluorescence_indices.xlsx ({exc})')
        if 'fri' in wanted:
            try:
                zf.writestr('FRI.xlsx', fri_xlsx_bytes(dataset))
            except Exception as exc:  # noqa: BLE001
                skipped.append(f'FRI.xlsx ({exc})')
        if 'slope' in wanted:
            if abs_obj is None:
                skipped.append('slope_fit_result.xlsx (no absorbance data)')
            else:
                zf.writestr('slope_fit_result.xlsx', slope_xlsx_bytes(abs_obj))
        if skipped:
            zf.writestr('skipped.txt', '\n'.join(skipped) + '\n')
    return buf.getvalue()
