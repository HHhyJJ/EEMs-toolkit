from __future__ import annotations

import os
import pandas as pd
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter
from typing import Optional, Sequence, Iterable
from pathlib import Path
from .absorbance import Abs
from . import export as _export
from . import parafac as _parafac
from .project import take_snapshot, apply_snapshot
from .constant import WATER_FREQ, CROSS, PEAK_A, PEAK_B, PEAK_C, PEAK_D, PEAK_M, PEAK_N, PEAK_T, PEAKS, \
    RAMAN_EM_RANGE, RAMAN_EX, FLUI, FREI, HIX, BIX

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid  # type: ignore


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
            file_list: Optional[np.ndarray[tuple[int], np.dtype[np.str_]]] | Sequence[str] = None,
    ):
        self.x = np.array(x, dtype=np.float64, copy=True)
        self.nSample = self.x.shape[0]
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
        self.blanked = False
        self.ife = False

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
        mask = np.zeros((self.nem, self.nex), dtype=np.bool_)
        if first[0] != 0 or first[1] != 0:
            mask |= (em_col >= ex_row - first[0]) & (em_col <= ex_row + first[1])
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
        mask = np.zeros((self.nem, self.nex), dtype=np.bool_)
        if first[0] != 0 or first[1] != 0:
            mask |= (em_col >= ram_row - first[0]) & (em_col <= ram_row + first[1])
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
            self.x[j] = griddata((self.ex[nan_ed[1]], self.em[nan_ed[0]]), i[nan_ed], (x, y), method='cubic')  # type: ignore
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
            ex_orders: Optional[np.ndarray[tuple[int], np.dtype[np.bool_]]] = None,
            em_orders: Optional[np.ndarray[tuple[int], np.dtype[np.bool_]]] = None,
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
        if self.blanked:
            print("Blank has been applied.")
            return
        if blank.ndim == 2:
            blank = np.array([blank for _ in range(self.nSample)])
        self.x -= blank
        self.x[self.x < 0] = 0
        self.blanked = True
        print('Blank minus done.')

    def dilute(self, dilution_factors: Iterable[float]):
        """
        Dilution correction in place.

        :param dilution_factors: A list of dilution factors for every sample.
        """
        # for i in range(self.nSample):
        #     self.x[i] *= dilution_factors[i]
        dilution_factors = np.array(dilution_factors, dtype=np.float64)
        self.x = np.einsum('ijk,i->ijk', self.x, dilution_factors)
        print('Dilution correction done.')

    def inner_effect_correct(self, absorbance: Abs):
        """
        Inner filter effect correction in place.
            Reference DOI: 10.1021/es0155276, 10.4319/lom.2013.11.616

        :param absorbance: An Abs object.
        """
        if self.ife:
            print("Inner filter effect correction has been applied.")
            return
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
        self.ife = True
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
        em_low, em_high = float(em_range[0]), float(em_range[1])
        if em_low >= em_high:
            raise ValueError('em_range must be an increasing (low, high) pair.')
        if self.nem < 2:
            raise ValueError('The emission axis is too short for Raman normalisation.')
        if self.em.min() > em_low or self.em.max() < em_high:
            raise ValueError(
                f'em_range {tuple(em_range)} is not covered by the emission axis '
                f'({float(self.em.min())}, {float(self.em.max())}).'
            )
        blank = np.asarray(blank, dtype=np.float64)
        if blank.ndim == 2:
            blank = np.repeat(blank[None, ...], self.nSample, axis=0)
        if blank.shape != (self.nSample, self.nem, self.nex):
            raise ValueError(
                f'Blank shape {blank.shape} does not match the dataset '
                f'({self.nSample}, {self.nem}, {self.nex}).'
            )
        em = np.arange(em_range[0], em_range[1] + 0.5, 0.5)
        ex_index = int(np.argmin(np.abs(self.ex - ex)))
        em_index = (int(np.where(self.em <= em_low)[0][-1]),
                    int(np.where(self.em >= em_high)[0][0]))
        if em_index[0] >= em_index[1]:
            raise ValueError('The emission axis does not span the Raman integration range.')
        for i in range(self.nSample):
            areal = blank[i, em_index[0]:em_index[1] + 1, ex_index]
            if not np.isfinite(areal).all():
                raise ValueError('The blank contains missing values inside the Raman integration range.')
            f = interp1d(self.em[em_index[0]:em_index[1] + 1], areal)
            f_new = f(em)
            RA = trapezoid(f_new, x=em)
            if not np.isfinite(RA) or RA == 0:
                raise ValueError('The Raman area is zero or not finite; check the blank sample.')
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
        ex = range(int(np.ceil(self.ex.min())), int(np.floor(self.ex.max())) + 1, 1)
        em = range(int(np.ceil(self.em.min())), int(np.floor(self.em.max())) + 1, 1)
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
            fitted = griddata((self.ex[nan_ed[1]], self.em[nan_ed[0]]), i[nan_ed], (x, y), method='linear')  # type: ignore
            if checked['B']:
                _peaks['B'].append(fitted[em_index[PEAK_B[0]], ex_index[PEAK_B[1]]])
            if checked['T']:
                _peaks['T'].append(fitted[em_index[PEAK_T[0]], ex_index[PEAK_T[1]]])
            if checked['A']:
                _peaks['A'].append(fitted[em_index[PEAK_A[0][0]]:em_index[PEAK_A[0][1]] + 1, ex_index[PEAK_A[1]]].max())
            if checked['M']:
                _peaks['M'].append(fitted[em_index[PEAK_M[0][0]]:em_index[PEAK_M[0][1]] + 1, ex_index[PEAK_M[1][0]]:ex_index[PEAK_M[1][1]] + 1].max())
            if checked['C']:
                _peaks['C'].append(fitted[em_index[PEAK_C[0][0]]:em_index[PEAK_C[0][1]], ex_index[PEAK_C[1][0]]:ex_index[PEAK_C[1][1]] + 1].max())
            if checked['D']:
                _peaks['D'].append(
                    fitted[em_index[PEAK_D[0]], ex_index[PEAK_D[1]]])
            if checked['N']:
                _peaks['N'].append(
                    fitted[em_index[PEAK_N[0]], ex_index[PEAK_N[1]]])

            if checked['FluI']:
                scan370 = savgol_filter(fitted[:, ex_index[FLUI[1]]], 21, 2)
                fl.append(scan370[em_index[FLUI[0][0]]] / scan370[em_index[FLUI[0][1]]])
            if checked['FreI']:
                scan310 = savgol_filter(fitted[:, ex_index[FREI[1]]], 21, 2)
                fre.append(scan310[em_index[FREI[0][0]]] / scan310[em_index[FREI[0][1][0]]:em_index[FREI[0][1][1]] + 1].max())  # type: ignore
            if checked['HIX']:
                scan254 = savgol_filter(fitted[:, ex_index[HIX[1]] + 1], 21, 2)
                v1 = scan254[em_index[HIX[0][0][0]]:em_index[HIX[0][0][1]] + 1].sum()  # type: ignore
                v2 = scan254[em_index[HIX[0][1][0]]:em_index[HIX[0][1][1]] + 1].sum()  # type: ignore
                if hix_type == 0:
                    hix.append(v1 / (v1 + v2))
                else:
                    hix.append(v1 / v2)
            if checked['BIX']:
                scan310 = savgol_filter(fitted[:, ex_index[FREI[1]]], 21, 2)
                bix.append(scan310[em_index[BIX[0][0]]] / scan310[em_index[BIX[0][1]]])
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
        ex = np.arange(np.ceil(self.ex.min()), np.floor(self.ex.max()) + 0.5, 0.5)
        em = np.arange(np.ceil(self.em.min()), np.floor(self.em.max()) + 0.5, 0.5)
        xx, yy = np.meshgrid(ex, em)
        new_x = np.zeros((self.nSample, em.shape[0], ex.shape[0]), dtype=np.float64)
        # nan_ed = np.where(~np.isnan(self.x[0]))
        for j, i in enumerate(self.x):
            nan_ed = np.where(~np.isnan(i))
            new_x[j] = griddata((self.ex[nan_ed[1]], self.em[nan_ed[0]]), i[nan_ed], (xx, yy), method='linear')  # type: ignore
        new_x[new_x < 0] = 0
        mask = ex[None, :] > em[:, None]
        new_x[:, mask] = 0
        s = np.zeros((5,))
        s[1] = (250 - ex.min()) * (380 - 330)
        s[2] = (250 - ex.min()) * (em.max() - 380)
        if em.min() < 250:
            if ex.min() < em.min():
                s[0] = (250 - ex.min()) * (330 - em.min()) - 0.5 * (250 - em.min()) ** 2
            else:
                s[0] = 0.5 * (330 - 250 + 330 - ex.min()) * (250 - ex.min())
            if ex.max() >= 380:
                s[3] = 0.5 * (380 - 250) ** 2
                s[4] = (ex.max() - 250) * (em.max() - 380) - 0.5 * (ex.max() - 380) ** 2
            else:
                s[3] = 0.5 * (ex.max() - 250) * (380 - ex.max() + 380 - 250)
                s[4] = (ex.max() - 250) * (em.max() - 380)
        else:
            s[0] = (250 - ex.min()) * (330 - em.min())
            if ex.max() >= 380:
                s[3] = 0.5 * (em.min() - 250 + 380 - 250) * (380 - em.min())
                s[4] = (ex.max() - 250) * (em.max() - 380) - 0.5 * (ex.max() - 380) ** 2
            else:
                s[3] = (em.min() - 250) * (380 - em.min()) + 0.5 * (ex.max() - em.min()) * (380 - ex.max() + 380 - em.min())
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
                ts = t[index[j][1][1]:index[j][1][0] + 1, index[j][0][1]:index[j][0][0] + 1]
                phi[i, j] = trapezoid(
                    trapezoid(
                        ts,
                        ex[index[j][0][1]:index[j][0][0] + 1], axis=1
                    ),
                    em[index[j][1][1]:index[j][1][0] + 1]
                )
                p[i, j] = phi[i, j] * s[j]
        p = (p.T / p.sum(axis=1).T).T
        s = s.reshape((1, 5))
        return p, phi, s

    #  Parallel Factors Analysis
    def multi_non_parafac_cal(
            self, f: Optional[int | Iterable[int]] = None,
            tol: float = 1e-6,
            max_iter: int = 2500,
            starts: int = 10,
            *,
            n_processes: Optional[int] = None,
            progress=None,
            stop_event=None,
    ):
        """
        Run 'start' times Non-negative CP decomposition via HALS. Uses Hierarchical ALS (Alternating Least Squares)
        which updates each factor column-wise (one column at a time while keeping all other
        columns fixed)

        :param f: int | Iterable, numbers of components
        :param tol: float, optional, the algorithm stops when the variation in the reconstruction
            error is less than the tol. Default: 1e-6
        :param max_iter: int, maximum number of iteration
        :param starts: int, number of models to fit
        :param n_processes: Optional, number of worker processes, defaults to min(starts, 8).
        :param progress: Optional, callback called as progress(done, total, f).
        :param stop_event: Optional, threading.Event used to cancel the job.
        """
        if f is None:
            print('Parameter f is not given, use range(2, 6) instead.')
        fs = _parafac.component_numbers(f, default=range(2, 6))
        print("Fitting started.")
        result = _parafac.fit_models(
            self.x, fs,
            starts=starts, tol=tol, max_iter=max_iter,
            n_processes=n_processes, progress=progress, stop_event=stop_event,
        )
        for line in result.log:
            print(line)
        print("All fits end.")
        self.fac_nums = list(fs)
        self.factors = result.factors
        self.leverage = result.leverage
        self.sse = result.sse
        self.core_consistency = result.core_consistency
        self.explanation_rate = result.explanation_rate

    def sse_analysis(self, f: Optional[int | Iterable[int]] = None) -> dict[int, dict]:
        """
        Compute the sum of squared errors (SSE) for fitted models.

        The squared error tensor is aggregated along each of the three axes:
        per sample, per emission wavelength and per excitation wavelength.

        :param f: int | Iterable, numbers of components. Defaults to the fitted models.
        :return: {f: {'total': float, 'sample': ndarray, 'em': ndarray, 'ex': ndarray}}
        """
        fs = _parafac.component_numbers(f, default=self.fac_nums if self.fac_nums else range(2, 6))
        fs = [fi for fi in fs if fi in self.factors]
        if not fs:
            raise ValueError('No fitted PARAFAC model is available.')
        result = {}
        for fi in fs:
            model = tl.cp_to_tensor(self.factors[fi])
            squared_error = (model - self.x) ** 2
            squared_error[np.isnan(squared_error)] = 0
            result[fi] = {
                'total': float(squared_error.sum()),
                'sample': squared_error.sum(axis=(1, 2)),
                'em': squared_error.sum(axis=(0, 2)),
                'ex': squared_error.sum(axis=(0, 1)),
            }
        return result

    def plot_sse(self, f: Optional[int | Iterable[int]] = None):
        """
        Plot SSE for one or more component numbers.

        The first panel shows the total SSE against the number of components, the
        other three show the SSE aggregated per sample, per emission and per
        excitation wavelength so that different component numbers can be compared.

        :param f: int | Iterable, numbers of components. Defaults to the fitted models.
        """
        data = self.sse_analysis(f)
        fs = sorted(data)
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(fs)))
        _ = plt.figure(figsize=(13, 8))
        # fig.canvas.manager.set_window_title('SSE analysis')
        ax = plt.subplot(2, 2, 1)
        ax.plot(fs, [data[i]['total'] for i in fs], '-o', color='#1f77b4')
        ax.set_yscale('log')
        ax.set_xticks(fs)
        ax.set_xlabel('Number of component', fontsize=11)
        ax.set_ylabel('Total SSE', fontsize=11)
        ax.set_title('Total SSE', fontsize=13)
        panels = (
            ('sample', 'SSE by sample', 'Sample', np.arange(1, self.nSample + 1)),
            ('em', 'SSE by emission wavelength', 'Emission/nm', self.em),
            ('ex', 'SSE by excitation wavelength', 'Excitation/nm', self.ex),
        )
        for index, (key, title, xlabel, xvalues) in enumerate(panels, start=2):
            ax = plt.subplot(2, 2, index)
            for color, fi in zip(colors, fs):
                ax.plot(xvalues, data[fi][key], color=color, lw=1.5, label=f'{fi} components')
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('SSE', fontsize=11)
            ax.set_title(title, fontsize=13)
            ax.legend(framealpha=0.6, fontsize=10, frameon=True, shadow=False)
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.25, hspace=0.35)

    def split_analysis(
            self, f: Optional[int | Iterable[int]] = None,
            tol: float = 1e-6,
            max_iter: int = 2500,
            starts: int = 10,
            random_state: Optional[int] = None,
            *,
            n_processes: Optional[int] = None,
            progress=None,
            stop_event=None,
    ):
        """
        Generate PARAFAC models nested in dataset splits. TCC above 0.95 can be validated.
            Reference DOI: 10.1027/1614-2241.2.2.57

        :param f: int, number of components
        :param random_state: {None(default), int, np.random.RandomState}
        :param starts: int, number of models to fit
        :param n_processes: Optional, number of worker processes, defaults to min(starts, 8).
        :param progress: Optional, callback called as progress(done, total, f).
        :param stop_event: Optional, threading.Event used to cancel the job.
        """
        if f is None:
            print('Parameter f is not given, use range(2, 6) instead.')
        fs = _parafac.component_numbers(f, default=self.fac_nums if self.fac_nums else range(2, 6))
        print('Split analysis starts.')
        result = _parafac.split_fit(
            self.x, self.factors, fs,
            starts=starts, tol=tol, max_iter=max_iter, random_state=random_state,
            n_processes=n_processes, progress=progress, stop_event=stop_event,
        )
        for line in result.log:
            print(line)
        print("All fits end.")
        self.split_result = result.split_result

    def split_validation(self, f: Optional[int | Iterable[int]] = None):
        if f is None:
            print('Parameter f is not given, use range(2, 6) instead.')
        fs = _parafac.component_numbers(f, default=self.fac_nums if self.fac_nums else range(2, 6))
        print("Split Validation Starts.")
        report = _parafac.split_validation(self.factors, self.split_result, fs)
        for fi in fs:
            print(f'Split Validation for {fi}-component:')
            if report[fi]['validated']:
                print(f"Validated for all comparisons of {fi}-component.")
            else:
                print(
                    f"Not Validated for {fi}-component, the lowest TCC is {report[fi]['min_tcc']:.4f}")
        return report

    #  State snapshots
    def snapshot(self) -> dict:
        """Return a deep copy of the mutable state (used by the history timeline)."""
        return take_snapshot(self)

    def restore(self, snapshot: dict):
        """Restore the mutable state in place from :meth:`snapshot` output."""
        apply_snapshot(self, snapshot)

    def copy(self) -> 'EEMsDataset':
        """Return an independent copy of this dataset."""
        new = EEMsDataset(
            self.x.copy(), self.ex.copy(), self.em.copy(), self.file_list.copy()
        )
        new.x = self.x.copy()
        new.i = self.i.copy()
        new.iu = self.iu
        new.blanked = self.blanked
        new.ife = self.ife
        new.fac_nums = list(self.fac_nums)
        new.factors = dict(self.factors)
        new.core_consistency = dict(self.core_consistency)
        new.explanation_rate = dict(self.explanation_rate)
        new.sse = dict(self.sse)
        new.leverage = dict(self.leverage)
        new.split_result = dict(self.split_result)
        return new

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
                cs = ax.contourf(x, y, model[i])
                fig.colorbar(cs, ax=ax)
            ax.set_xlabel('Excitation/nm', fontsize=11)
            ax.set_ylabel('Emission/nm', fontsize=11)
            if f is None:
                ax.set_title(
                    f'Sample {i + 1}: {self.file_list[i]}', fontsize=13)
            else:
                ax.set_title(
                    f'Model {f} for sample {i + 1}', fontsize=13)
            plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)

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
            plt.axvline(330, ymax=(250 - self.ex.min()) / (self.ex.max() - self.ex.min()), color='k', linestyle='--', )
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
            plt.annotate(f'{self.i[i] + 1}: {self.file_list[i]}', (self.leverage[f][i], self.sse[f][i]))
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
            err = self.x[i] - model[i]
            plt.contourf(self.ex, self.em, err)
            plt.xlabel('Excitation/nm', fontsize=11)
            plt.ylabel('Emission/nm', fontsize=11)
            plt.title(f'Residual', fontsize=13)
            plt.colorbar()
            plt.subplots_adjust(left=0.07, bottom=0.13, right=0.94, top=0.9, wspace=0.24, hspace=0.2)

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
        plt.subplots_adjust(left=0.07, bottom=0.13, right=0.94, top=0.9, wspace=0.24, hspace=0.2)

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
            axes[0, ind].set_title(f'Model {f}\n{title} - Em')
            axes[1, ind].set_title(f'Model {f}\n{title} - Ex')
            for jix in range(f):
                line, = axes[0, ind].plot(self.em, self.split_result[f][2 * ind][1][1][:, jix],
                                          ls='-', label=f'1st comp {jix + 1}')
                axes[0, ind].plot(self.em, self.split_result[f][2 * ind + 1][1][1][:, jix],
                                  ls='--', label=f'2nd comp {jix + 1}', c=line.get_color())
                axes[0, ind].plot(self.em, self.factors[f][1][1][:, jix],
                                  ls=':', label=f'Overall comp {jix + 1}', c=line.get_color())
                line, = axes[1, ind].plot(self.ex, self.split_result[f][2 * ind][1][2][:, jix],
                                          ls='-', label=f'1st comp {jix + 1}')
                axes[1, ind].plot(self.ex, self.split_result[f][2 * ind + 1][1][2][:, jix],
                                  ls='--', label=f'2nd comp {jix + 1}', c=line.get_color())
                axes[1, ind].plot(self.ex, self.factors[f][1][2][:, jix],
                                  ls=':', label=f'Overall comp {jix + 1}', c=line.get_color())
        lines, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(lines, labels, framealpha=0.6, loc='center right', fontsize=11, frameon=True, shadow=False, bbox_to_anchor=(1.0, 0.5))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.83, top=0.87, wspace=0.3, hspace=0.47)

    def plot_fmax(self, f: int, use_file_list: bool = False):
        """
        Plot FMax to see components' variation among samples.
        :param f: int, number of components
        :param use_file_list: Optional, if True, use file_list in xticks. Default is False
        """
        _ = plt.figure(figsize=(6, 4.8))
        # fig.canvas.manager.set_window_title(f'FMax plot of {f}-component')
        fmax = self.factors[f][1][0] * self.factors[f][1][1].max(axis=0) * self.factors[f][1][2].max(axis=0)
        for i in range(f):
            plt.plot(fmax[:, i], label=f'Component {i + 1}')
        plt.xlabel('Sample', fontsize=11)
        if use_file_list:
            plt.xticks(range(self.nSample), self.file_list)  # type: ignore
        plt.ylabel('FMax', fontsize=11)
        plt.title('FMax for the samples', fontsize=13)
        plt.legend(framealpha=0.6, loc='upper right', fontsize=11, frameon=True, shadow=False)
        plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)

    def plot_fingers(self, f: int):
        """
        Plot components in one figure. It will print peaks of each component for it's Ex and Em.

        :param f: int, number of components
        """
        print(f'Description of PARAFAC components - Model {f}')
        peak = [[], []]
        for i in [1, 2]:
            w = self.em if i == 1 else self.ex
            for j in range(f):
                index = find_peaks(self.factors[f][1][i][:, j])
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
            x_ = np.outer(
                self.factors[f][1][1][:, ii - 1],
                self.factors[f][1][2][:, ii - 1]
            )
            plt.contourf(self.ex, self.em, x_)
            plt.title(f'Component {ii}', fontsize=13)
            if ii % c == 1:
                plt.ylabel('Emission/nm', fontsize=11)
            if ii > (r - 1) * c:
                plt.xlabel('Excitation/nm', fontsize=11)
        plt.subplots_adjust(left=left, bottom=bottom, right=0.94, top=top, wspace=0.24, hspace=0.2)

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
            plt.plot(self.ex, self.factors[f][1][2][:, ii - 1], label='Ex')
            plt.plot(self.em, self.factors[f][1][1][:, ii - 1], label='Em')
            plt.legend(framealpha=0.6, loc='upper right', fontsize=11, frameon=True, shadow=False)
            plt.xlim(min(self.ex.min(), self.em.min()), max(self.ex.max(), self.em.max()))
            plt.ylim(0)
            plt.title(f'Component {ii}', fontsize=13)
            if ii % c == 1:
                plt.ylabel('Loadings', fontsize=11)
            if ii > (r - 1) * c:
                plt.xlabel('$\lambda$/nm', fontsize=11)  # type: ignore
        plt.subplots_adjust(left=left, bottom=bottom, right=0.94, top=top, wspace=0.24, hspace=0.2)

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
        content = _export.open_fluor_text(self, f, **kwargs)
        with open(write_path, 'wb') as fp:
            fp.write(content)
        print(f"Saved at {str(write_path)}")

    def eems_output(self, output_path: Optional[str] = None):
        """
        Output the EEMs

        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        for i in range(self.nSample):
            if output_path is None:
                file_name = Path(os.getcwd()) / f"{self.file_list[i]}.xlsx"
            else:
                file_name = Path(output_path) / f"{self.file_list[i]}.xlsx"
            with open(file_name, 'wb') as fp:
                fp.write(_export.eem_xlsx_bytes(self, i))
            print(f'EEM {self.file_list[i]} is saved at {str(file_name)}. {i + 1} of {self.nSample}')
        print('All saved.')

    def parafac_result_output(self, f: int, output_path: Optional[str] = None, filename: str = 'parafac-result.xlsx'):
        """
        Output PARAFAC results, including FMax, Em loadings, Ex loadings and components.

        :param f: int, number of components
        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        if output_path is None:
            file_name = Path(os.getcwd()) / filename
        else:
            file_name = Path(output_path) / filename
        with open(file_name, 'wb') as fp:
            fp.write(_export.parafac_xlsx_bytes(self, f))
        print(f'PARAFAC result saved at {str(file_name)}')

    def fri_result_output(self, output_path: Optional[str] = None, filename: str = 'FRI.xlsx'):
        """
        Calculate FRI and output

        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        if output_path is None:
            file_name = Path(os.getcwd()) / filename
        else:
            file_name = Path(output_path) / filename
        with open(file_name, 'wb') as fp:
            fp.write(_export.fri_xlsx_bytes(self))
        print(f'FRI result saved at {str(file_name)}')

    def fluorescence_indices_output(self, output_path: Optional[str] = None, output_name: str = 'fluorescence_indices.xlsx'):
        """
        Calculate fluorescence indices and output

        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        if output_path is None:
            file_name = Path(os.getcwd()) / output_name
        else:
            file_name = Path(output_path) / output_name
        with open(file_name, 'wb') as fp:
            fp.write(_export.indices_xlsx_bytes(self))
        print(f'Flu Indices result saved at {str(file_name)}')
