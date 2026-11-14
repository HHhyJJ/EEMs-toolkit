# -*- coding = utf-8 -*-
# @TIME : 2023/01/10 12:29
# @File : EEMs_toolkit.py
# @Software : PyCharm
from __future__ import annotations

import os
import datetime
import tlviz
import pandas as pd
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter
from tensorly.decomposition import non_negative_parafac_hals
from tensorly.cp_tensor import CPTensor
from typing import Optional, Tuple, List, Union, Sequence, Iterable, overload
from multiprocessing.pool import Pool
from pathlib import Path

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

# slope fit
S_275_295 = (275, 295)
S_350_400 = (350, 400)
LONG_RANGE = (300, 600)
# flu indices
PEAKS = 'BTAMCDN'
PEAK_B = (305, 275)
PEAK_T = (340, 275)
PEAK_A = ((400, 460), 260)
PEAK_M = ((370, 410), (290, 310))
PEAK_C = ((420, 460), (320, 360))
PEAK_D = (509, 390)
PEAK_N = (370, 280)
FLUI = ((470, 520), 370)
FREI = ((380, (420, 430)), 310)
HIX = (((435, 480), (300, 345)), 254)
BIX = ((380, 430), 310)
# Raman Unit
RAMAN_EX = 350
RAMAN_EM_RANGE = (378, 424)
WATER_FREQ = 3382
# Split Analysis
N_SPLIT = 4
CROSS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


#  read files


def read_eems(sample_log: pd.DataFrame, eem_path: str) -> EEMsDataset:
    """
    Read the EEM samples with different Ex in columns and different Em in rows.
    The header is Ex and the index is Em.

    :param sample_log: A DataFrame with the column 'eem' in it.
    :param eem_path: The directory where the EEM files saved.
    :return: EEMsDataset.
    """
    file_list: list[str] = list(sample_log['eem'])
    nums = len(file_list)
    root = Path(eem_path)
    x = []
    try:
        for j, i in enumerate(file_list):
            file_path: Path = root / i
            if not file_path.exists():
                raise FileNotFoundError(str(file_path))
            c = pd.read_excel(file_list, header=None)
            x.append(c.values[1:, 1:])
            print(f'EEM {i} has been read. {j + 1} of {nums}')
        ex = c.values[0, 1:]
        em = c.values[1:, 0]
    except Exception as e:
        raise e

    fl = [i.split('.')[0] for i in file_list]
    try:
        x = np.array(x)
        return EEMsDataset(x, ex, em, fl)
    except Exception as e:
        raise e


def read_abs(sample_log: pd.DataFrame, abs_path: str) -> Abs:
    """
    Read the Abs samples.

    :param sample_log:  A DataFrame with the column 'abs' in it.
    :param abs_path: The directory where the Abs files saved.
    :return: A matrix of absorbance, and the wavelength of the absorbance.
    """
    file_list: set[str] = set(sample_log['abs'])
    nums = len(file_list)
    root = Path(abs_path)
    d = dict()
    try:
        for j, i in enumerate(file_list):
            file_path = root / i
            if not file_path.exists():
                raise FileNotFoundError(str(file_path))
            c = pd.read_excel(file_path, header=None)
            print(f'Abs {i} has been read. {j + 1} of {nums}')
            t = c.values
            if t.shape[0] == 2:
                t = t.T
            d[i] = t[:, 1]
        wave = t[:, 0]
        data = np.array([d[i] for i in sample_log['abs']])
        return Abs(wave, data, sample_log['abs'])
    except Exception as e:
        raise e


def read_blank(sample_log: pd.DataFrame, blank_path: str) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """
    Read the blank EEM samples with different Ex in columns and different Em in rows.
    The header is Ex and the index is Em.

    :param sample_log: A DataFrame with the column 'blank' in it.
    :param blank_path: The directory where the blank EEMs files saved.
    :return: blank_eem
    """
    file_list: set[str] = set(sample_log['blank'])
    nums = len(file_list)
    root = Path(blank_path)
    d = dict()
    try:
        for j, i in enumerate(file_list):
            file_path = root / i
            if not file_path.exists():
                raise FileNotFoundError(str(file_path))
            c = pd.read_excel(file_path, header=None)
            print(f'Blank {i} has been read. {j + 1} of {nums}')
            d[i] = c.values[1:, 1:]
        blank_eem = np.array([d[i] for i in sample_log['blank']])
        return blank_eem
    except Exception as e:
        raise e


def read_sample_log(path: str) -> pd.DataFrame:
    """
    Read the sample log. The example of sample log can be downloaded from GitHub.

    :param path: The directory where the sample log saved.
    :return: A DataFrame of sample log.
    """
    log = pd.read_excel(path, header=0, index_col=None)
    print('Sample log has been read.')
    return log


class EEMsDataset:
    """
    The object of EEMs dataset.
    """

    #  Initialization
    def __init__(
            self,
            x: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
            ex: np.ndarray[tuple[int], np.dtype[np.float64]],
            em: np.ndarray[tuple[int], np.dtype[np.float64]],
            file_list: Optional[np.ndarray[tuple[int],
                                           np.dtype[np.str_]]] | Sequence[str] = None,
    ):
        self.x = np.asarray(x)
        self.nSample = x.shape[0]
        self.ex = ex
        self.em = em
        self.nem = self.em.shape[0]
        self.nex = self.ex.shape[0]
        self.i = np.arange(self.nSample)
        if file_list is None:
            self.file_list = self.i.copy()
        else:
            self.file_list = np.array(file_list)
        mask = self.ex[None, :] > self.em[:, None]  # (nem, nex)
        self.x[:, mask] = 0
        # self.backup = x.copy()
        self.fac_nums: list[int] = []
        self.factors: dict[int, tuple] = dict()
        # self.weights = dict()
        self.core_consistency: dict[int, float] = dict()
        self.explanation_rate: dict[int, float] = dict()
        self.sse = dict()
        self.leverage = dict()
        self.split_result = dict()
        self.iu = 'AU'

    #  Preprocess Method
    def cut_ray_scatter(self, first: tuple[float, float], second: tuple[float, float]):
        """
        Remove Rayleigh scatter in place, use nans instead.

        :param first: A tuple with two elements, representing widths below and above 1st Rayleigh scatter center.
        :param second: A tuple with two elements, representing widths below and above 2nd Rayleigh scatter center.
        """
        if (first[0] == 0 and first[1] == 0) and (second[0] == 0 and second[1] == 0):
            return
        ex_row = self.ex[None, :]  # (1, nex)
        em_col = self.em[:, None]  # (nem, 1)
        mask = np.zeros((self.nem, self.nex), dtype=bool)
        if first[0] != 0 or first[1] != 0:
            mask |= (em_col >= ex_row - first[0]
                     ) & (em_col <= ex_row + first[1])
        if second[0] != 0 or second[1] != 0:
            ex2 = 2.0 * ex_row
            mask |= (em_col >= ex2 - second[0]) & (em_col <= ex2 + second[1])
        if mask.any():
            self.x[:, mask] = np.nan
        print('Cut Rayleigh Scatter done.')

    def cut_ram_scatter(self, first: tuple[float, float], second: tuple[float, float], freq: float = WATER_FREQ):
        """
        Remove Raman scatter in place, use nans instead.

        :param first: A tuple with two elements, representing widths below and above 1st Raman scatter center.
        :param second: A tuple with two elements, representing widths below and above 2nd Raman scatter center.
        :param freq: Optional, a constant named Raman shift which is around 3400-3600 cm^-1 for water. Default is 3382.
        """
        if (first[0] == 0 and first[1] == 0) and (second[0] == 0 and second[1] == 0):
            return
        ram = 1.0 / (1.0 / self.ex - freq / 1e7)  # (nex,)
        ram_row = ram[None, :]  # (1, nex)
        em_col = self.em[:, None]  # (nem, 1)
        mask = np.zeros((self.nem, self.nex), dtype=bool)
        if first[0] != 0 or first[1] != 0:
            mask |= (em_col >= ram_row -
                     first[0]) & (em_col <= ram_row + first[1])
        if second[0] != 0 or second[1] != 0:
            ram2 = 2.0 * ram_row
            mask |= (em_col >= ram2 - second[0]) & (em_col <= ram2 + second[1])
        if mask.any():
            self.x[:, mask] = np.nan
        print('Cut Raman Scatter done.')

    def miss_value_interpolation(self):
        """
        Use Delaunay triangulation method to interpolate in place, aim at removing nans in EEMs so
        that the parafac program can run normally.
            Reference DOI: 10.1016/j.marchem.2004.02.006
        """
        x, y = np.meshgrid(self.ex, self.em)
        for j, i in enumerate(self.x):
            nan_ed = np.where(~np.isnan(i.astype('float')))
            self.x[j] = griddata(
                (self.ex[nan_ed[1]], self.em[nan_ed[0]]), i[nan_ed], (x, y), method='cubic')
        self.x[self.x < 0] = 0
        print('Interpolation done.')

    def smooth_eem(self, sigma: float = 0.5, filter_size: Optional[int] = None):
        """
        Use Gaussian kernel to smooth EEMs in place. Suggesting after interpolation.

        :param sigma: Optional, standard deviation of the Gaussian distribution, specified as a positive number.
            Default is 0.5.
        :param filter_size: optional, size of the Gaussian filter, specified as a positive,
            odd integer. The default filter size is 2*ceil(2*sigma)+1.
        """
        if sigma <= 0:
            return
        if filter_size is None:
            filter_size = 2 * int(np.ceil(2 * sigma)) + 1
        center = filter_size // 2
        if center == 0:
            return
        truncate = (filter_size - 1) / (2.0 * sigma)
        self.x = gaussian_filter(
            self.x, sigma=(0, sigma, sigma),
            truncate=truncate, mode='mirror',
        )
        print('EEM smooth done.')

    def sub_dataset(
            self,
            sample_orders: Optional[list[int]] = None,
            ex_orders: Optional[np.ndarray[tuple[int],
                                           np.dtype[np.bool_]]] = None,
            em_orders: Optional[np.ndarray[tuple[int],
                                           np.dtype[np.bool_]]] = None,
    ):
        """
        Removing samples and/or wavelengths in place.

        :param sample_orders: A list of positive integers, 1 is the first sample.
        :param ex_orders: e.g. self.ex == 240
        :param em_orders: e.g. self.em > 600
        """
        if sample_orders:
            rest = np.full(self.i.shape, True)
            for i in sample_orders:
                ind = np.where(self.i == i - 1)[0]
                rest[ind] = False
            self.i = self.i[rest]
            self.file_list = self.file_list[rest]
            self.x = self.x[rest]
            self.nSample = self.x.shape[0]
        if ex_orders is not None:
            self.ex = self.ex[~ex_orders]
            self.nex = self.ex.shape[0]
            self.x = self.x[:, :, ~ex_orders]
        if em_orders is not None:
            self.em = self.em[~em_orders]
            self.nem = self.em.shape[0]
            self.x = self.x[:, ~em_orders, :]

    def minus_the_blank(
            self,
            blank: np.ndarray[tuple[int, int] | tuple[int, int, int], np.dtype[np.float64]],
    ):
        """
        Minus blank sample in place.

        :param blank: The same shape as self.x or just one sample.
        """
        if blank.ndim == 2:
            blank = np.array([blank for _ in range(self.nSample)])
        self.x -= blank
        self.x[self.x < 0] = 0
        print('Blank minus done.')

    def dilute(self, dilution_factors: np.ndarray[tuple[int], np.dtype[np.float64]]):
        """
        Dilution correction in place.

        :param dilution_factors: A list of dilution factors for every sample.
        """
        # for i in range(self.nSample):
        #     self.x[i] *= dilution_factors[i]
        self.x = np.einsum('ijk,i->ijk', self.x, dilution_factors)
        print('Dilution correction done.')

    def inner_effect_correct(self, absorbance: Abs):
        """
        Inner filter effect correction in place.
            Reference DOI: 10.1021/es0155276, 10.4319/lom.2013.11.616

        :param absorbance: An Abs object.
        """
        for ex in self.ex:
            if not absorbance.check_wavelength(ex):
                print(f'Ex wavelength {ex} is not included')
                return
        for em in self.em:
            if not absorbance.check_wavelength(em):
                print(f'Em wavelength {em} is not included')
                return
        for iSample in range(self.nSample):
            em_abs = absorbance(iSample, self.em)
            ex_abs = absorbance(iSample, self.ex)
            correct = 0.5 * (em_abs[:, None] + ex_abs[None, :])  # nem x nex
            self.x[iSample] *= np.power(10, correct)
        print('Inner effect correction done.')

    def raman_areal(self, blank: np.ndarray[tuple[int, int] | tuple[int, int, int], np.dtype[np.float64]],
                    ex: float = RAMAN_EX, em_range: tuple[float, float] = RAMAN_EM_RANGE):
        """
        Turning arbitrary units(AU) into Raman units(RU).
        Reference DOI: 10.1366/000370209788964548

        :param blank: The same shape as self.x or just one sample.
        :param ex: Optional, the excitation position of Raman peak. Default is 350.
        :param em_range: Optional, a tuple of two elements, range used to calculate the area,
            if not given, it will be (378, 424).
        """
        em = np.arange(em_range[0], em_range[1] + 0.5, 0.5)
        ex_index = np.argmin(np.abs(self.ex - ex))
        em_index = (np.where((self.em - em_range[0]) <= 0)[0][-1],
                    np.where((self.em - em_range[1]) >= 0)[0][0])
        if blank.ndim == 2:
            blank = np.array([blank for _ in range(self.nSample)])
        for i in range(self.nSample):
            areal = blank[i, em_index[0]:em_index[1] + 1, ex_index]
            f = interp1d(self.em[em_index[0]:em_index[1] + 1], areal)
            f_new = f(em)
            RA = trapezoid(f_new, x=em)
            self.x[i] /= RA
        self.iu = 'RU'
        print('Raman Units done.')

    #  Fluorescence Indices
    def _pick_fluorescence_indices(self, hix_type: int = 0) -> pd.DataFrame:
        """
        Calculate fluorescence indices.
            Fluorescence index: Ratio of em wavelengths at 470 nm and 520 nm, obtained at ex370
                Reference DOI: 10.1016/j.gca.2006.06.1554
            Freshness index: em 380 divided by the em maximum between 420 and 435, obtained at ex310
                Reference DOI: 10.1016/S0146-6380(00)00124-8
            Humification index: area under the em spectra 435:480 divided by the peak area 300:345 + 435:480, at ex254
                Reference DOI: 10.1016/S0146-6380(00)00124-8
            Biological index: Ex 310, Em 380 divided by 430
                Reference DOI: 10.1016/j.orggeochem.2009.03.002

        :param hix_type: Optional, 0 means v1/(v1+v2) else v1/v2. Default is 0.
        """
        ex = range(int(np.ceil(self.ex.min())),
                   int(np.floor(self.ex.max())) + 1, 1)
        em = range(int(np.ceil(self.em.min())),
                   int(np.floor(self.em.max())) + 1, 1)
        ex_index = {i: j for j, i in enumerate(ex)}
        em_index = {i: j for j, i in enumerate(em)}
        checked = {i: False for i in PEAKS}
        if PEAK_B[0] in em and PEAK_B[1] in ex:
            checked['B'] = True
        if PEAK_T[0] in em and PEAK_T[1] in ex:
            checked['T'] = True
        if PEAK_A[0][0] in em and PEAK_A[0][1] in em and PEAK_A[1] in ex:
            checked['A'] = True
        if PEAK_M[0][0] in em and PEAK_M[0][1] in em and PEAK_M[1][0] in ex and PEAK_M[1][1] in ex:
            checked['M'] = True
        if PEAK_C[0][0] in em and PEAK_C[0][1] in em and PEAK_C[1][0] in ex and PEAK_C[1][1] in ex:
            checked['C'] = True
        if PEAK_D[0] in em and PEAK_D[1] in ex:
            checked['D'] = True
        if PEAK_N[0] in em and PEAK_N[1] in ex:
            checked['N'] = True
        if FLUI[0][0] in em and FLUI[0][1] in em and FLUI[1] in ex:
            checked['FluI'] = True
        else:
            checked['FluI'] = False
        if FREI[0][0] in em and FREI[0][1][0] in em and FREI[0][1][1] in em and FREI[1] in ex:
            checked['FreI'] = True
        else:
            checked['FreI'] = False
        if HIX[0][0][0] in em and HIX[0][0][1] in em and HIX[0][1][0] in em and HIX[0][1][1] in em and HIX[1] in ex:
            checked['HIX'] = True
        else:
            checked['HIX'] = False
        if BIX[0][0] in em and BIX[0][1] in em and BIX[1] in ex:
            checked['BIX'] = True
        else:
            checked['BIX'] = False
        x, y = np.meshgrid(ex, em)
        letter = PEAKS
        _peaks = {i: [] for i in letter}
        fl, fre, hix, bix = [], [], [], []
        for i in self.x:
            nan_ed = np.where(~np.isnan(i))
            fitted = griddata((self.ex[nan_ed[1]], self.em[nan_ed[0]]), i[nan_ed],
                              (x, y), method='linear')
            if checked['B']:
                _peaks['B'].append(
                    fitted[em_index[PEAK_B[0]], ex_index[PEAK_B[1]]])
            if checked['T']:
                _peaks['T'].append(
                    fitted[em_index[PEAK_T[0]], ex_index[PEAK_T[1]]])
            if checked['A']:
                _peaks['A'].append(fitted[em_index[PEAK_A[0][0]]:em_index[PEAK_A[0][1]] + 1,
                                   ex_index[PEAK_A[1]]].max())
            if checked['M']:
                _peaks['M'].append(fitted[em_index[PEAK_M[0][0]]:em_index[PEAK_M[0][1]] + 1,
                                   ex_index[PEAK_M[1][0]]:ex_index[PEAK_M[1][1]] + 1].max())
            if checked['C']:
                _peaks['C'].append(fitted[em_index[PEAK_C[0][0]]:em_index[PEAK_C[0][1]],
                                   ex_index[PEAK_C[1][0]]:ex_index[PEAK_C[1][1]] + 1].max())
            if checked['D']:
                _peaks['D'].append(
                    fitted[em_index[PEAK_D[0]], ex_index[PEAK_D[1]]])
            if checked['N']:
                _peaks['N'].append(
                    fitted[em_index[PEAK_N[0]], ex_index[PEAK_N[1]]])

            if checked['FluI']:
                scan370 = savgol_filter(fitted[:, ex_index[FLUI[1]]], 21, 2)
                fl.append(scan370[em_index[FLUI[0][0]]] /
                          scan370[em_index[FLUI[0][1]]])
            if checked['FreI']:
                scan310 = savgol_filter(fitted[:, ex_index[FREI[1]]], 21, 2)
                fre.append(
                    scan310[em_index[FREI[0][0]]] / scan310[em_index[FREI[0][1][0]]:em_index[FREI[0][1][1]] + 1].max())
            if checked['HIX']:
                scan254 = savgol_filter(fitted[:, ex_index[HIX[1]] + 1], 21, 2)
                v1 = scan254[em_index[HIX[0][0][0]]
                    :em_index[HIX[0][0][1]] + 1].sum()
                v2 = scan254[em_index[HIX[0][1][0]]
                    :em_index[HIX[0][1][1]] + 1].sum()
                if hix_type == 0:
                    hix.append(v1 / (v1 + v2))
                else:
                    hix.append(v1 / v2)
            if checked['BIX']:
                scan310 = savgol_filter(fitted[:, ex_index[FREI[1]]], 21, 2)
                bix.append(scan310[em_index[BIX[0][0]]] /
                           scan310[em_index[BIX[0][1]]])
        out = pd.DataFrame()
        for i in letter:
            if checked[i]:
                out[i] = _peaks[i]
        if checked['FluI']:
            out['FluI'] = fl
        if checked['FreI']:
            out['FreI'] = fre
        if checked['HIX']:
            out['HIX'] = hix
        if checked['BIX']:
            out['BIX'] = bix
        out.insert(loc=0, column='Sample', value=self.file_list)
        return out

    #  FRI
    def _fluorescence_regional_integration(self) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], ...]:
        """
        Calculation of FRI, including percent fluorescence response, volume of different regions
        and multiplication factor.
            Reference DOI: 10.1021/es034354c
        """
        if self.ex.min() >= 250:
            raise ValueError("Regions 1, 2, and 3 do not exist!")
        if self.ex.max() <= 250:
            raise ValueError("Regions 4 and 5 do not exist!")
        if self.em.min() >= 380:
            raise ValueError("Regions 1, 2, and 4 do not exist!")
        elif self.em.min() >= 330:
            raise ValueError("Regions 1 does not exist!")
        if self.em.max() <= 330:
            raise ValueError("Regions 2, 3, and 5 do not exist!")
        elif self.em.max() <= 380:
            raise ValueError("Regions 3 and 5 do not exist!")
        ex = np.arange(np.ceil(self.ex.min()),
                       np.floor(self.ex.max()) + 0.5, 0.5)
        em = np.arange(np.ceil(self.em.min()),
                       np.floor(self.em.max()) + 0.5, 0.5)
        xx, yy = np.meshgrid(ex, em)
        new_x = np.zeros(
            (self.nSample, em.shape[0], ex.shape[0]), dtype=np.float64)
        # nan_ed = np.where(~np.isnan(self.x[0]))
        for j, i in enumerate(self.x):
            nan_ed = np.where(~np.isnan(i))
            new_x[j] = griddata((self.ex[nan_ed[1]], self.em[nan_ed[0]]), i[nan_ed],  # noqa
                                (xx, yy), method='linear')
        new_x[new_x < 0] = 0
        for i, ii in enumerate(ex):
            for j, jj in enumerate(em):
                if ii >= jj:
                    new_x[:, j, i] = 0
        s = np.zeros((5,))
        s[1] = (250 - ex.min()) * (380 - 330)
        s[2] = (250 - ex.min()) * (em.max() - 380)
        if em.min() < 250:
            if ex.min() < em.min():
                s[0] = (250 - ex.min()) * (330 - em.min()) - \
                    0.5 * (250 - em.min()) ** 2
            else:
                s[0] = 0.5 * (330 - 250 + 330 - ex.min()) * (250 - ex.min())
            if ex.max() >= 380:
                s[3] = 0.5 * (380 - 250) ** 2
                s[4] = (ex.max() - 250) * (em.max() - 380) - \
                    0.5 * (ex.max() - 380) ** 2
            else:
                s[3] = 0.5 * (ex.max() - 250) * (380 - ex.max() + 380 - 250)
                s[4] = (ex.max() - 250) * (em.max() - 380)
        else:
            s[0] = (250 - ex.min()) * (330 - em.min())
            if ex.max() >= 380:
                s[3] = 0.5 * (em.min() - 250 + 380 - 250) * (380 - em.min())
                s[4] = (ex.max() - 250) * (em.max() - 380) - \
                    0.5 * (ex.max() - 380) ** 2
            else:
                s[3] = (em.min() - 250) * (380 - em.min()) + \
                    0.5 * (ex.max() - em.min()) * \
                       (380 - ex.max() + 380 - em.min())
                s[4] = (ex.max() - 250) * (em.max() - 380)
        index = [[[np.where(ex == 250)[0][0], np.argmin(ex)],
                  [np.where(em == 330)[0][0], np.argmin(em)]],
                 [[np.where(ex == 250)[0][0], np.argmin(ex)],
                  [np.where(em == 380)[0][0], np.where(em == 330)[0][0]]],
                 [[np.where(ex == 250)[0][0], np.argmin(ex)],
                  [np.argmax(em), np.where(em == 380)[0][0]]],
                 [[np.argmax(ex), np.where(ex == 250)[0][0]],
                  [np.where(em == 380)[0][0], np.argmin(em)]],
                 [[np.argmax(ex), np.where(ex == 250)[0][0]],
                  [np.argmax(em), np.where(em == 380)[0][0]]]]
        s = s.sum() / s
        p, phi = np.zeros((self.nSample, 5)), np.zeros((self.nSample, 5))
        for i in range(self.nSample):
            t = new_x[i]
            for j in range(5):
                ts = t[index[j][1][1]:index[j][1][0] +
                       1, index[j][0][1]:index[j][0][0] + 1]
                phi[i, j] = trapezoid(trapezoid(ts, ex[index[j][0][1]:index[j][0][0] + 1], axis=1),
                                      em[index[j][1][1]:index[j][1][0] + 1])
                p[i, j] = phi[i, j] * s[j]
        p = (p.T / p.sum(axis=1).T).T
        s = s.reshape((1, 5))
        return p, phi, s

    @overload
    @staticmethod
    def _non_parafac_once(x, f: int, tol: float = 1e-6, max_iter: int = 2500, init: str = 'random', *, j: None = None) -> tuple[CPTensor | tuple, int]:
        ...

    @overload
    @staticmethod
    def _non_parafac_once(x, f: int, tol: float = 1e-6, max_iter: int = 2500, init: str = 'random', *, j: int) -> tuple[CPTensor | tuple, int, int]:
        ...

    #  Parallel Factors Analysis
    @staticmethod
    def _non_parafac_once(x, f: int, tol: float = 1e-6, max_iter: int = 2500, init: str = 'random', *, j: Optional[int] = None) -> tuple[CPTensor | tuple, int, int] | tuple[CPTensor | tuple, int]:
        r = non_negative_parafac_hals(
            tensor=tl.tensor(x),
            rank=f,
            n_iter_max=max_iter,
            init=init, tol=tol,
            return_errors=False,
        )
        print(f"A {f}-component model has been fitted.")
        if j is not None:
            return r, f, j
        else:
            return r, f

    def multi_non_parafac_cal(
        self, f: Optional[int | Iterable[int]] = None,
        tol: float = 1e-6,
        max_iter: int = 2500,
        starts: int = 10,
    ):
        """
        Run 'start' times Non-negative CP decomposition via HALS. Uses Hierarchical ALS (Alternating Least Squares)
        which updates each factor column-wise (one column at a time while keeping all other
        columns fixed)

        :param f: int | Iterable, numbers of components
        :param tol: float, optional, the algorithm stops when the variation in the reconstruction
            error is less than the tol. Default: 1e-6
        :param max_iter: int, maximum number of iteration
        :param start: int, number of models to fit
        """
        if np.any(np.isnan(self.x)):
            raise ValueError("x contains nan.")
        if f is None:
            fs = range(2, 6)
            print('Parameter f is not given, use range(2, 6) instead.')
        elif isinstance(f, int):
            fs = (f,)
        else:
            fs = f
        num_cores = os.cpu_count()
        if num_cores is not None:
            if num_cores >= starts:
                n = starts
            else:
                n = 8
        else:
            n = 8
        print("Fitting started.")
        results: dict[int, list[CPTensor | tuple]] = {i: [] for i in fs}

        def callback(rtn: tuple[CPTensor | tuple, int]):
            results[rtn[1]].append(rtn[0])
        pool = Pool(n)
        for fi in fs:
            for _ in range(starts):
                pool.apply_async(
                    self._non_parafac_once,
                    args=(self.x, fi, tol, max_iter),
                    callback=callback,
                    error_callback=lambda e: print(e)
                )
        pool.close()
        pool.join()
        print("All fits end.")
        # postprocess
        post_results: dict[int, tuple] = {}
        self.fac_nums = list(fs)
        self.leverage.clear()
        self.sse.clear()
        self.core_consistency.clear()
        self.explanation_rate.clear()
        for fi in results:
            res_i = results[fi]
            best_model = tlviz.multimodel_evaluation.get_model_with_lowest_error(
                res_i, self.x)
            best_model = tlviz.factor_tools.distribute_weights_in_one_mode(
                best_model, mode=0)
            best_model = tlviz.factor_tools.permute_cp_tensor(best_model)
            post_results[fi] = best_model
            model = tl.cp_tensor.cp_to_tensor(best_model)
            self.leverage[fi] = tlviz.outliers.compute_leverage(
                best_model[1][0])
            self.sse[fi] = tlviz.outliers.compute_slabwise_sse(
                model, self.x, normalise=True)
            self.core_consistency[fi] = tlviz.model_evaluation.core_consistency(
                best_model, self.x, normalised=True)
            self.explanation_rate[fi] = 1 - tlviz.model_evaluation.relative_sse(
                best_model, self.x)
        self.factors = post_results

    def split_analysis(
        self, f: Optional[int | Iterable[int]] = None,
        tol: float = 1e-6,
        max_iter: int = 2500,
        starts: int = 10,
        random_state: Optional[int] = None,
    ):
        """
        Generate PARAFAC models nested in dataset splits. TCC above 0.95 can be validated.
            Reference DOI: 10.1027/1614-2241.2.2.57

        :param f: int, number of components
        :param random_state: {None(default), int, np.random.RandomState}
        :param start: int, number of models to fit
        """
        if np.any(np.isnan(self.x)):
            raise ValueError("x contains nan.")
        if f is None:
            fs = self.fac_nums if self.fac_nums else range(2, 6)
            print('Parameter f is not given, use range(2, 6) instead.')
        elif isinstance(f, int):
            fs = (f,)
        else:
            fs = f
        num_cores = os.cpu_count()
        if num_cores is not None:
            if num_cores >= starts:
                n = starts
            else:
                n = 8
        else:
            n = 8

        print('Split analysis starts.')
        index = np.arange(self.nSample)
        rng = np.random.RandomState(random_state)
        rng.shuffle(index)
        cn = int(N_SPLIT * (N_SPLIT - 1) / 1 / 2)  # C(4, 2)
        index_split = [index[i * self.nSample //
                             N_SPLIT:(i + 1) * self.nSample // N_SPLIT] for i in range(N_SPLIT)]
        splits = [self.x[i] for i in index_split]
        results = {i: [[] for _ in range(cn)] for i in fs}

        def callback1(rtn: tuple[CPTensor | tuple, int, int]):
            results[rtn[1]][rtn[2]*2].append(rtn[0])

        def callback2(rtn: tuple[CPTensor | tuple, int, int]):
            results[rtn[1]][rtn[2]*2+1].append(rtn[0])
        pool = Pool(n)
        for j, (n1, n2) in enumerate(CROSS):
            c_x1 = np.concatenate((splits[n1[0]], splits[n1[1]]), axis=0)
            c_x2 = np.concatenate((splits[n2[0]], splits[n2[1]]), axis=0)
            for fi in fs:
                for _ in range(starts):
                    pool.apply_async(
                        self._non_parafac_once,
                        args=(c_x1, fi, tol, max_iter),
                        kwds={'j': j},
                        callback=callback1,
                        error_callback=lambda e: print(e),
                    )
                    pool.apply_async(
                        self._non_parafac_once,
                        args=(c_x2, fi, tol, max_iter),
                        kwds={'j': j},
                        callback=callback2,
                        error_callback=lambda e: print(e),
                    )
        pool.close()
        pool.join()
        print("All fits end.")
        # postprocess
        self.split_result.clear()
        for fi in fs:
            for ji, jj in enumerate(results[fi]):
                c1, c2 = divmod(ji, 2)
                x = np.concatenate(
                    (splits[CROSS[c1][c2][0]], splits[CROSS[c1][c2][1]]), axis=0)
                best_model = tlviz.multimodel_evaluation.get_model_with_lowest_error(
                    jj, x)
                best_model = tlviz.factor_tools.distribute_weights_in_one_mode(
                    best_model, mode=0)
                order = tlviz.factor_tools.get_factor_matrix_permutation(
                    self.factors[fi][1][1], best_model[1][1])
                best_model = tlviz.factor_tools.permute_cp_tensor(
                    best_model, permutation=tuple(order))
                results[fi][ji] = best_model  # type: ignore
        self.split_result = results

    def split_validation(self, f: Optional[int | Iterable[int]] = None):
        if f is None:
            fs = self.fac_nums if self.fac_nums else range(2, 6)
            print('Parameter f is not given, use range(2, 6) instead.')
        elif isinstance(f, int):
            fs = (f,)
        else:
            fs = f
        print("Split Validation Starts.")
        for fi in fs:
            print(f'Split Validation for {fi}-component:')
            ex_l = [
                tlviz.factor_tools.cosine_similarity(
                    self.factors[fi][1][2], ti[1][2]
                )
                for ti in self.split_result[fi]
            ]
            em_l = [
                tlviz.factor_tools.cosine_similarity(
                    self.factors[fi][1][1], ti[1][1]
                )
                for ti in self.split_result[fi]
            ]
            for i in range(len(CROSS)):
                ex_l.append(
                    tlviz.factor_tools.cosine_similarity(
                        self.split_result[fi][2*i][1][2],
                        self.split_result[fi][2*i+1][1][2]
                    )
                )
                em_l.append(
                    tlviz.factor_tools.cosine_similarity(
                        self.split_result[fi][2*i][1][1],
                        self.split_result[fi][2*i+1][1][1]
                    )
                )
            m1 = min(ex_l)
            m2 = min(em_l)
            mm = min(m1, m2)
            if mm >= 0.95:
                print(f"Validated for all comparisons of {fi}-component.")
            else:
                print(
                    f"Not Validated for {fi}-component, the lowest TCC is {mm:.4f}")

    #  Plot Functions

    def plot_eem_by1(self, f: Optional[int] = None, sample_id: Optional[np.typing.ArrayLike] = None, plot_3d: bool = False):
        """
        Plot EEMs

        :param f: Optional, if given, plot model from factors instead self.x. Default is None.
        :param sample_id: Optional, which samples to plot, if not given, plot all. Default is None.
        :param plot_3d: Optional, if True then plot 3D else contourf. Default is False.
        """
        if sample_id is None:
            sample_id = range(self.nSample)
        else:
            sample_id = np.array(sample_id, dtype=np.int32) - 1
        if plot_3d:
            x, y = np.meshgrid(self.ex, self.em)
        else:
            x, y = self.ex, self.em
        if f is not None:
            model = tl.cp_to_tensor(self.factors[f])
        else:
            model = self.x
        for i in sample_id:
            fig = plt.figure(figsize=(6, 4.8))
            # fig.canvas.manager.set_window_title('Plot Raw EEMs')  # type: ignore
            if plot_3d:
                ax = fig.add_subplot(projection='3d')
                ax.plot_surface(x, y, model[i])
            else:
                ax = fig.add_subplot()
                ax.contourf(x, y, model[i])
            ax.set_xlabel('Excitation/nm', fontsize=11)
            ax.set_ylabel('Emission/nm', fontsize=11)
            if f is None:
                ax.set_title(
                    f'Sample {i + 1}: {self.file_list[i]}', fontsize=13)
            else:
                ax.set_title(
                    f'Model {f} for sample {i + 1}', fontsize=13)
            plt.subplots_adjust(left=0.13, bottom=0.11,
                                right=0.95, top=0.9)

    def plot_fri(self, sample_id: Optional[Sequence[int]] = None):
        """
        Plot FRI
            Reference DOI: 10.1021/es034354c
        """
        if sample_id is None:
            ls = range(self.nSample)
        else:
            ls = np.asarray(sample_id, dtype=np.int32) - 1
        for i in ls:
            _ = plt.figure(figsize=(6, 4.8))
            # fig.canvas.manager.set_window_title('Plot Fluorescence Region') # type: ignore
            plt.contourf(self.em, self.ex, self.x[i].T)
            plt.axvline(380, color='k', linestyle='--')
            plt.axhline(250, color='k', linestyle='--')
            plt.axvline(330, ymax=(250 - self.ex.min()) /
                                  (self.ex.max() - self.ex.min()), color='k', linestyle='--', )
            plt.text((self.em.min() + 330) / 2, (self.ex.min() + 250) / 2, 'Ⅰ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.text((380 + 330) / 2, (self.ex.min() + 250) / 2, 'Ⅱ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.text((self.em.max() + 380) / 2, (self.ex.min() + 250) / 2, 'Ⅲ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.text((self.em.min() + 380) / 2, (self.ex.max() + 250) / 2, 'Ⅳ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.text((self.em.max() + 380) / 2, (self.ex.max() + 250) / 2, 'Ⅴ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.ylabel('Excitation/nm', fontsize=11)
            plt.xlabel('Emission/nm', fontsize=11)
            plt.title(f'Sample {i + 1}: {self.file_list[i]}', fontsize=13)
            plt.colorbar()
            plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)

    def plot_outlier_test(self, f: int):
        """
        Create a new figure to show outliers.

        :param f: int, number of components
        """
        _ = plt.figure(figsize=(6, 4.8))
        # ax = plt.axes()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # fig.canvas.manager.set_window_title(f'Plot the outlier test of {f}-component model')
        plt.scatter(self.leverage[f], self.sse[f])
        for i in range(len(self.sse[f])):
            plt.annotate(f'{self.i[i] + 1}: {self.file_list[i]}',
                         (self.leverage[f][i], self.sse[f][i]))
        plt.xlabel('Leverage', fontsize=11)
        plt.ylabel('SSE', fontsize=11)
        plt.title(f'Outlier plot for {f} components', fontsize=13)
        plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)

    def plot_residual_error(self, f: int, sample_id: Optional[np.typing.ArrayLike] = None):
        """
        Plot sample, model from factors and residual in one figure.

        :param f: int, number of component
        :param sample_id: Optional, which samples to plot, if not given, plot all. Default is None.
        """
        if sample_id is None:
            sample_id = range(self.nSample)
        else:
            sample_id = np.array(sample_id, dtype=np.int32) - 1
        model = tl.cp_to_tensor(self.factors[f])
        for i in sample_id:
            _ = plt.figure(figsize=(4.5 * 3, 4))
            # fig.canvas.manager.set_window_title('Plot model and residual')
            plt.subplot(1, 3, 1)
            plt.contourf(self.ex, self.em, self.x[i])
            plt.xlabel('Excitation/nm', fontsize=11)
            plt.ylabel('Emission/nm', fontsize=11)
            plt.title(f'Sample {i + 1}: {self.file_list[i]}', fontsize=13)
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.contourf(self.ex, self.em, model[i])
            plt.xlabel('Excitation/nm', fontsize=11)
            plt.ylabel('Emission/nm', fontsize=11)
            plt.title(f'Model {f} for Sample {i + 1}', fontsize=13)
            plt.colorbar()
            plt.subplot(1, 3, 3)
            err = self.x[i] - model[f]
            plt.contourf(self.ex, self.em, err)
            plt.xlabel('Excitation/nm', fontsize=11)
            plt.ylabel('Emission/nm', fontsize=11)
            plt.title(f'Residual', fontsize=13)
            plt.colorbar()
            plt.subplots_adjust(left=0.07, bottom=0.13,
                                right=0.94, top=0.9, wspace=0.24, hspace=0.2)

    def plot_core_consistency_and_explanation(self):
        """
        Plot core consistency and explanation rate, with f in self.f
        """
        _ = plt.figure(figsize=(9, 4.3))
        # fig.canvas.manager.set_window_title('Core consistency and explanation rate')
        fs = sorted(self.fac_nums)
        ep = [100 * self.explanation_rate[i] for i in fs]
        cc = [self.core_consistency[i] for i in fs]
        plt.subplot(1, 2, 1)
        plt.title('Core consistency', fontsize=13)
        plt.plot(fs, cc, '-o')
        plt.xlabel('Number of component', fontsize=11)
        # plt.ylabel('%', fontsize=11)
        plt.xticks(fs)
        print(f'Core consistency: {self.core_consistency}')
        plt.subplot(1, 2, 2)
        plt.title('Explanation rate', fontsize=13)
        plt.plot(fs, ep, '-o')
        plt.xlabel('Number of component', fontsize=11)
        plt.ylabel('%', fontsize=11)
        plt.xticks(fs)
        print(f'Explanation rate: {self.explanation_rate}')
        plt.subplots_adjust(left=0.07, bottom=0.13, right=0.94,
                            top=0.9, wspace=0.24, hspace=0.2)

    def plot_split_result(self, f: int):
        """
        Plot split analysis result

        :param f: int, number of components
        """
        fig = plt.figure(figsize=(13, 6))
        # fig.canvas.manager.set_window_title(f'Split analysis result for {f}-component')
        axes = fig.subplots(2, len(CROSS))
        ts = 'ABCD'
        for ind in range(len(CROSS)):
            title = ''.join([ts[xc] for xc in CROSS[ind][0]]) + ' vs ' + \
                ''.join([ts[xc] for xc in CROSS[ind][1]])
            axes[0, ind].seet_title(f'Model {f}\n{title} - Em')
            axes[1, ind].seet_title(f'Model {f}\n{title} - Ex')
            for jix in range(f):
                line, = axes[0, ind].plot(self.em, self.split_result[f][2*ind][1][1][:, jix],
                                          ls='--', label=f'1st comp {jix+1}')
                axes[0, ind].plot(self.em, self.split_result[f][2*ind+1][1][1][:, jix],
                                  ls='--', label=f'2nd comp {jix+1}', c=line.get_color())
                axes[0, ind].plot(self.em, self.factors[f][1][1][:, jix],
                                  ls=':', label=f'Overall comp {jix+1}', c=line.get_color())
                line, = axes[1, ind].plot(self.em, self.split_result[f][2*ind][1][2][:, jix],
                                          ls='--', label=f'1st comp {jix+1}')
                axes[1, ind].plot(self.em, self.split_result[f][2*ind+1][1][2][:, jix],
                                  ls='--', label=f'2nd comp {jix+1}', c=line.get_color())
                axes[1, ind].plot(self.em, self.factors[f][1][2][:, jix],
                                  ls=':', label=f'Overall comp {jix+1}', c=line.get_color())
        lines, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(lines, labels, framealpha=0.6, loc='center right',
                   fontsize=11, frameon=True, shadow=False, bbox_to_anchor=(1.0, 0.5))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.83,
                            top=0.87, wspace=0.3, hspace=0.47)

    def plot_fmax(self, f: int, use_file_list: bool = False):
        """
        Plot FMax to see components' variation among samples.
        :param f: int, number of components
        :param use_file_list: Optional, if True, use file_list in xticks. Default is False
        """
        _ = plt.figure(figsize=(6, 4.8))
        # fig.canvas.manager.set_window_title(f'FMax plot of {f}-component')
        fmax = self.factors[f][0] * \
            self.factors[f][1].max(axis=0) * self.factors[f][2].max(axis=0)
        for i in range(f):
            plt.plot(fmax[:, i], label=f'Component {i + 1}')
        plt.xlabel('Sample', fontsize=11)
        if use_file_list:
            plt.xticks(range(self.nSample), self.file_list)  # type: ignore
        plt.ylabel('FMax', fontsize=11)
        plt.title('FMax for the samples', fontsize=13)
        plt.legend(framealpha=0.6, loc='upper right',
                   fontsize=11, frameon=True, shadow=False)
        plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)

    def plot_fingers(self, f: int):
        """
        Plot components in one figure. It will print peaks of each component for it's Ex and Em.

        :param f: int, number of components
        """
        model = tl.cp_to_tensor(self.factors[f])
        print(f'Description of PARAFAC components - Model {f}')
        peak = [[], []]
        for i in [1, 2]:
            w = self.em if i == 1 else self.ex
            for j in range(f):
                index = find_peaks(self.factors[f][i][:, j])
                peak[i - 1].append(w[index[0]])
        for j in range(f):
            print(f'Component {j + 1}')
            for i in [1, 2]:
                if i == 1:
                    print('Em:', end='\t')
                else:
                    print('Ex:', end='\t')
                for k in peak[i - 1][j]:
                    print(k, end='\t')
                print('', end='\n')
        r, c, rr = 1, f, 4.3
        left, bottom, top = 0.07, 0.13, 0.9
        if f > 3:
            r, c, rr = 2, int(np.ceil(f / 2)), 7.5
            left, bottom, top = 0.09, 0.076, 0.92
        _ = plt.figure(figsize=(4.5 * c, rr))
        # fig.canvas.manager.set_window_title('Plot components')
        for ii in range(1, f + 1):
            plt.subplot(r, c, ii)
            plt.contourf(self.ex, self.em, model[ii - 1])
            plt.title(f'Component {ii}', fontsize=13)
            if ii % c == 1:
                plt.ylabel('Emission/nm', fontsize=11)
            if ii > (r - 1) * c:
                plt.xlabel('Excitation/nm', fontsize=11)
        plt.subplots_adjust(left=left, bottom=bottom,
                            right=0.94, top=top, wspace=0.24, hspace=0.2)

    def plot_loadings(self, f: int):
        """
        Plot loadings of components

        :param f: int, number of components
        """
        r, c, rr = 1, f, 4.3
        left, bottom, top = 0.07, 0.13, 0.9
        if f > 3:
            r, c, rr = 2, int(np.ceil(f / 2)), 7.5
            left, bottom, top = 0.09, 0.076, 0.92
        _ = plt.figure(figsize=(4.5 * c, rr))
        # fig.canvas.manager.set_window_title('Plot loadings for all components')
        for ii in range(1, f + 1):
            plt.subplot(r, c, ii)
            plt.plot(self.ex, self.factors[f][2][:, ii - 1], label='Ex')
            plt.plot(self.em, self.factors[f][1][:, ii - 1], label='Em')
            plt.legend(framealpha=0.6, loc='upper right',
                       fontsize=11, frameon=True, shadow=False)
            plt.xlim(min(self.ex.min(), self.em.min()),
                     max(self.ex.max(), self.em.max()))
            plt.ylim(0)
            plt.title(f'Component {ii}', fontsize=13)
            if ii % c == 1:
                plt.ylabel('Loadings', fontsize=11)
            if ii > (r - 1) * c:
                plt.xlabel('$\lambda$/nm', fontsize=11)
        plt.subplots_adjust(left=left, bottom=bottom,
                            right=0.94, top=top, wspace=0.24, hspace=0.2)

    #  Data output
    def open_fluor(self, f: int, output_path: Optional[str] = None, filename: str = 'model.txt', **kwargs):
        """
        Output a text file which can be uploaded to the OpenFluor to compare.

        :param f: int, number of components
        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        :param kwargs: Optional, consist of personal information
        """
        if output_path is None:
            write_path = Path(os.getcwd()) / filename
        else:
            write_path = Path(output_path) / filename
        ex_str = ''
        for i in range(self.nex):
            ex_str += f'EX\t{self.ex[i]}\t'
            s = '\t'.join(f'{self.factors[f][2][i, j]}' for j in range(f))
            ex_str = ex_str + s + '\n'
        em_str = ''
        for i in range(self.nem):
            em_str += f'EM\t{self.em[i]}\t'
            s = '\t'.join(f'{self.factors[f][1][i, j]}' for j in range(f))
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
        with open(write_path, 'w') as fp:
            fp.write(f'#\n# Fluorescence Model\n#\n'
                     f'name\t{name}\ncreator\t{creator}\nemail\t{email}\ndoi\t{doi}\n'
                     f'reference\t{reference}\nunit\t{self.iu}\ntoolbox\tEEM toolkit for Python\n'
                     f'date\t{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
                     f'fluorometer\t{fluorometer}\nnSample\t{self.nSample}\nconstraints\tnon-negativity\n'
                     f'validation\tSplit-half analysis (S4C4T2)\nmethods\t{method}\n'
                     f'preprocess\t{preprocess}\nsources\t{sources}\necozones\t{ecozones}\n'
                     f'description\t{description}\n'
                     f'#\n# Excitation/Emission (Ex, Em), wavelength (nm), component[n] (intensity)\n#\n'
                     f'{ex_str}{em_str}')
        print(f"Saved at {str(write_path)}")

    def eems_output(self, output_path: Optional[str] = None):
        """
        Output the EEMs

        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        for i in range(self.nSample):
            out = np.full((self.nem + 1, self.nex + 1), np.nan)
            out[1:, 1:] = self.x[i]
            out[0, 1:] = self.ex
            out[1:, 0] = self.em
            out = pd.DataFrame(out)
            if output_path is None:
                file_name = Path(f"{self.file_list[i]}.xlsx'")
            else:
                file_name = Path(output_path) / f"{self.file_list[i]}.xlsx'"
            out.to_excel(file_name, header=False, index=False)
            print(
                f'EEM {self.file_list[i]} is saved at {str(file_name)}. {i + 1} of {self.nSample}')
        print('All saved.')

    def parafac_result_output(self, f: int, output_path: Optional[str] = None, filename: str = 'parafac-result.xlsx'):
        """
        Output PARAFAC results, including FMax, Em loadings, Ex loadings and components.

        :param f: int, number of components
        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        fmax = pd.DataFrame(self.factors[f][0] * self.factors[f][1].max(axis=0) * self.factors[f][2].max(axis=0),
                            columns=[f'FMax{i + 1}' for i in range(f)])
        fmax.insert(loc=0, column='EEM', value=self.file_list)
        em_loadings = pd.DataFrame(self.factors[f][1],
                                   columns=[f'Em{i + 1}' for i in range(f)])
        em_loadings.insert(loc=0, column='Em', value=self.em)
        ex_loadings = pd.DataFrame(self.factors[f][2],
                                   columns=[f'Ex{i + 1}' for i in range(f)])
        ex_loadings.insert(loc=0, column='Ex', value=self.ex)
        out_model = []
        for i in range(3):
            out = np.full((self.nem + 1, self.nex + 1), np.nan)
            j = np.outer(self.factors[f][1][1], self.factors[f][1][2])
            out[1:, 1:] = j
            out[0, 1:] = self.ex
            out[1:, 0] = self.em
            out_model.append(pd.DataFrame(out))
        if output_path is None:
            file_name = Path(filename)
        else:
            file_name = Path(output_path) / filename
        with pd.ExcelWriter(file_name) as writer:
            fmax.to_excel(writer, sheet_name='FMax', index=False)
            em_loadings.to_excel(writer, sheet_name='Em loadings', index=False)
            ex_loadings.to_excel(writer, sheet_name='Ex loadings', index=False)
            for j in range(len(out_model)):
                out_model[j].to_excel(
                    writer, sheet_name=f'C{j + 1}', header=False, index=False)
        print(f'PARAFAC result saved at {str(file_name)}')

    def fri_result_output(self, output_path: Optional[str] = None, filename: str = 'FRI.xlsx'):
        """
        Calculate FRI and output

        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        p, phi, mf = self._fluorescence_regional_integration()
        cols = np.arange(1, 6)
        pfr = pd.DataFrame(p, columns=cols)
        pfr.insert(loc=0, column='Sample', value=self.file_list)
        volume = pd.DataFrame(phi, columns=cols)
        volume.insert(loc=0, column='Sample', value=self.file_list)
        mf = pd.DataFrame(mf, columns=cols)
        if output_path is None:
            file_name = filename
        else:
            file_name = Path(output_path) / filename
        with pd.ExcelWriter(file_name) as writer:
            pfr.to_excel(
                writer, sheet_name='percent fluorescence response', index=False)
            volume.to_excel(writer, sheet_name='volume', index=False)
            mf.to_excel(
                writer, sheet_name='multiplication factor', index=False)
        print(f'FRI result saved at {str(file_name)}')

    def fluorescence_indices_output(self, output_path: Optional[str] = None,
                                    output_name: str = 'fluorescence_indices.xlsx'):
        """
        Calculate fluorescence indices and output

        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        df = self._pick_fluorescence_indices()
        if output_path is None:
            file_name = output_name
        else:
            file_name = Path(output_path) / output_name
        df.to_excel(file_name, index=False)
        print(f'Flu Indices result saved at {str(file_name)}')


class Abs:
    """
    The object of absorbance dataset
    """

    def __init__(self, wavelength: np.ndarray, absorbance: np.ndarray,
                 file_list: Sequence[str] | np.ndarray | pd.Series):
        """
        :param wavelength: The wavelengths of Abs.
        :param absorbance: A matrix with different samples in rows and different wavelengths in columns.
        :param file_list: File list of each sample.
        """

        sorted_abs_id = sorted(
            range(wavelength.shape[0]), key=lambda k: wavelength[k])
        self.wavelength = wavelength[sorted_abs_id]
        if absorbance.shape[1] == wavelength.shape[0]:
            self.absorbance = absorbance
        elif absorbance.shape[0] == wavelength.shape[0]:
            self.absorbance = absorbance.T
        else:
            return
        self.absorbance = self.absorbance[:, sorted_abs_id]
        self.interp = [interp1d(self.wavelength, self.absorbance[i, :])
                       for i in range(self.absorbance.shape[0])]
        self.file_list = np.array(file_list)

    def slope_fit(self, long_range: Tuple[int, int] = LONG_RANGE) -> pd.DataFrame:
        """
        Fit spectral slopes in two different areas of the CDOM absorbance spectrum.
        It will create a new excel file named 'slope_fit_result.xlsx' in the current directory.
        Exponential model: y = a350*exp(S/1000*(350-lambda))+k
            Reference DOI: 10.4319/lo.2008.53.3.0955, 10.4319/lo.2001.46.8.2087

        :param long_range: Wavelength range for the long wavelength slope. Default is (300, 600).
        :return: A DataFrame of results.
        """

        def fit_fun(x, *args):
            b1, b2, b3 = args
            return b1 * np.exp(b2 / 1000 * (350 - x)) + b3

        def r_squared(aa, y_true, ww):
            y_p = np.polyval(aa, ww)
            r2 = 1 - ((y_p - y_true) ** 2).sum() / \
                ((y_true - y_true.mean()) ** 2).sum()
            r2_adjust = 1 - (1 - r2) * (len(ww) - 1) / (len(ww) - len(aa))
            return r2_adjust

        results = {'exp_slope_microm': None,
                   'S_275_295': None, 'S_350_400': None, 'SR': None}
        if long_range[0] in self.wavelength and long_range[1] in self.wavelength:
            index = [np.where(long_range[0] == self.wavelength)[0][0],
                     np.where(long_range[1] == self.wavelength)[0][0]]
            ws = self.wavelength[index[0]:index[1] + 1]
            abc = self.absorbance[:, index[0]:index[1] + 1]
            r0 = [curve_fit(fit_fun, ws, abc[i, :], p0=[abc[i, :].mean(), 18, 0], maxfev=2500)[0]
                  for i in range(self.absorbance.shape[0])]
            fit0 = [(fit_fun(ws, *r0[i]) ** 2).sum() / (abc[i, :] ** 2).sum()
                    for i in range(self.absorbance.shape[0])]
            results['exp_slope_microm'] = ([x[1] for x in r0], fit0)
        if S_275_295[0] in self.wavelength and S_275_295[1] in self.wavelength:
            index = [np.where(S_275_295[0] == self.wavelength)[0][0],
                     np.where(S_275_295[1] == self.wavelength)[0][0]]
            ws = self.wavelength[index[0]:index[1] + 1]
            abc = np.log(self.absorbance[:, index[0]:index[1] + 1])
            p1 = [np.polyfit(ws, abc[i, :], 1)
                  for i in range(self.absorbance.shape[0])]
            fit = [r_squared(p1[i], abc[i, :], ws)
                   for i in range(self.absorbance.shape[0])]
            results['S_275_295'] = ([x[0] * -1e3 for x in p1], fit)
        if S_350_400[0] in self.wavelength and S_350_400[1] in self.wavelength:
            index = [np.where(S_350_400[0] == self.wavelength)[0][0],
                     np.where(S_350_400[1] == self.wavelength)[0][0]]
            ws = self.wavelength[index[0]:index[1] + 1]
            abc = np.log(self.absorbance[:, index[0]:index[1] + 1])
            p1 = [np.polyfit(ws, abc[i, :], 1)
                  for i in range(self.absorbance.shape[0])]
            fit = [r_squared(p1[i], abc[i, :], ws)
                   for i in range(self.absorbance.shape[0])]
            results['S_350_400'] = ([x[0] * -1e3 for x in p1], fit)
        if results['S_275_295'] is not None and results['S_350_400'] is not None:
            results['SR'] = np.array(results['S_275_295'][0]) / np.array(results['S_350_400'][0])  # noqa
        out = pd.DataFrame()
        # out['file_list'] = self.file_list
        if results['exp_slope_microm'] is not None:
            out['exp_slope_microm'] = results['exp_slope_microm'][0]
            out['r2_adjust_for_exp_slope_microm'] = results['exp_slope_microm'][1]
        if results['S_275_295'] is not None:
            out['S_275_295'] = results['S_275_295'][0]
            out['r2_adjust_for_S_275_295'] = results['S_275_295'][1]
        if results['S_350_400'] is not None:
            out['S_350_400'] = results['S_350_400'][0]
            out['r2_adjust_for_S_350_400'] = results['S_350_400'][1]
        if results['SR'] is not None:
            out['SR'] = results['SR']
        return out

    def check_wavelength(self, wave: Union[int, float]) -> bool:
        if self.wavelength[0] <= wave <= self.wavelength[-1]:
            return True
        else:
            return False

    def __call__(self, index: int, wave: np.ndarray) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return self.interp[index](wave)
