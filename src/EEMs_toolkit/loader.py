from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable
from .absorbance import Abs
from .eem import EEMsDataset


def read_eems(file_list: Iterable[str], eem_path: str) -> EEMsDataset:
    """
    Read the EEM samples with different Ex in columns and different Em in rows.
    The header is Ex and the index is Em.

    :param file_list: A list of eem filenames.
    :param eem_path: The directory where the EEM files saved.
    :return: EEMsDataset.
    """
    file_list = list(file_list)
    nums = len(file_list)
    root = Path(eem_path)
    x = []
    try:
        for j, i in enumerate(file_list):
            file_path: Path = root / i
            if not file_path.exists():
                raise FileNotFoundError(str(file_path))
            c = pd.read_excel(file_path, header=None)
            x.append(c.values[1:, 1:])
            print(f'EEM {i} has been read. {j + 1} of {nums}')
        ex = c.values[0, 1:]  # type: ignore
        em = c.values[1:, 0]  # type: ignore
    except Exception as e:
        raise e

    fl = [i.split('.')[0] for i in file_list]
    try:
        x = np.array(x)
        return EEMsDataset(x, ex, em, fl)
    except Exception as e:
        raise e


def read_abs(file_list: Iterable[str], abs_path: str) -> Abs:
    """
    Read the Abs samples.

    :param file_list:  A list of abs filenames.
    :param abs_path: The directory where the Abs files saved.
    :return: Absorbance dataset.
    """
    file_list = list(file_list)
    file_list_: set[str] = set(file_list)
    nums = len(file_list_)
    root = Path(abs_path)
    d = dict()
    try:
        for j, i in enumerate(file_list_):
            file_path = root / i
            if not file_path.exists():
                raise FileNotFoundError(str(file_path))
            c = pd.read_excel(file_path, header=None)
            print(f'Abs {i} has been read. {j + 1} of {nums}')
            t = c.values
            if t.shape[0] == 2:
                t = t.T
            d[i] = t[:, 1]
        wave = t[:, 0]  # type: ignore
        data = np.array([d[i] for i in file_list])
        return Abs(wave, data, file_list)
    except Exception as e:
        raise e


def read_blank(file_list: Iterable[str], blank_path: str) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """
    Read the blank EEM samples with different Ex in columns and different Em in rows.
    The header is Ex and the index is Em.

    :param file_list: A list of blank filenames.
    :param blank_path: The directory where the blank EEMs files saved.
    :return: blank_eem
    """
    file_list = list(file_list)
    file_list_: set[str] = set(file_list)
    nums = len(file_list_)
    root = Path(blank_path)
    d = dict()
    try:
        for j, i in enumerate(file_list_):
            file_path = root / i
            if not file_path.exists():
                raise FileNotFoundError(str(file_path))
            c = pd.read_excel(file_path, header=None)
            print(f'Blank {i} has been read. {j + 1} of {nums}')
            d[i] = c.values[1:, 1:]
        blank_eem = np.array([d[i] for i in file_list])
        return blank_eem
    except Exception as e:
        raise e
