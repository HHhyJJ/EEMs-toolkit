from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typing import Sequence, Any
from .constant import LONG_RANGE, S_275_295, S_350_400


class Abs:
    """
    The object of absorbance dataset
    """

    def __init__(self, wavelength: np.ndarray, absorbance: np.ndarray, file_list: Sequence[str]):
        """
        :param wavelength: The wavelengths of Abs.
        :param absorbance: A matrix with different samples in rows and different wavelengths in columns.
        :param file_list: File list of each sample.
        """

        sorted_abs_id = sorted(range(wavelength.shape[0]), key=lambda k: wavelength[k])
        self.wavelength = wavelength[sorted_abs_id]
        if absorbance.shape[1] == wavelength.shape[0]:
            self.absorbance = absorbance
        elif absorbance.shape[0] == wavelength.shape[0]:
            self.absorbance = absorbance.T
        else:
            return
        self.absorbance = self.absorbance[:, sorted_abs_id]
        self.interp = [interp1d(self.wavelength, self.absorbance[i, :]) for i in range(self.absorbance.shape[0])]
        self.file_list = np.array(file_list)

    def slope_fit(self, long_range: tuple[int, int] = LONG_RANGE) -> pd.DataFrame:
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

        results: dict[str, Any] = {'exp_slope_microm': None, 'S_275_295': None, 'S_350_400': None, 'SR': None}
        if long_range[0] in self.wavelength and long_range[1] in self.wavelength:
            index = [
                np.where(long_range[0] == self.wavelength)[0][0],
                np.where(long_range[1] == self.wavelength)[0][0]
            ]
            ws = self.wavelength[index[0]:index[1] + 1]
            abc = self.absorbance[:, index[0]:index[1] + 1]
            r0 = [curve_fit(fit_fun, ws, abc[i, :], p0=[abc[i, :].mean(), 18, 0], maxfev=2500)[0]
                  for i in range(self.absorbance.shape[0])]
            fit0 = [(fit_fun(ws, *r0[i]) ** 2).sum() / (abc[i, :] ** 2).sum()
                    for i in range(self.absorbance.shape[0])]
            results['exp_slope_microm'] = ([x[1] for x in r0], fit0)
        if S_275_295[0] in self.wavelength and S_275_295[1] in self.wavelength:
            index = [
                np.where(S_275_295[0] == self.wavelength)[0][0],
                np.where(S_275_295[1] == self.wavelength)[0][0]
            ]
            ws = self.wavelength[index[0]:index[1] + 1]
            abc = np.log(self.absorbance[:, index[0]:index[1] + 1])
            p1 = [np.polyfit(ws, abc[i, :], 1) for i in range(self.absorbance.shape[0])]
            fit = [r_squared(p1[i], abc[i, :], ws) for i in range(self.absorbance.shape[0])]
            results['S_275_295'] = ([x[0] * -1e3 for x in p1], fit)
        if S_350_400[0] in self.wavelength and S_350_400[1] in self.wavelength:
            index = [
                np.where(S_350_400[0] == self.wavelength)[0][0],
                np.where(S_350_400[1] == self.wavelength)[0][0]
            ]
            ws = self.wavelength[index[0]:index[1] + 1]
            abc = np.log(self.absorbance[:, index[0]:index[1] + 1])
            p1 = [np.polyfit(ws, abc[i, :], 1) for i in range(self.absorbance.shape[0])]
            fit = [r_squared(p1[i], abc[i, :], ws) for i in range(self.absorbance.shape[0])]
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

    def check_wavelength(self, wave: float) -> bool:
        if self.wavelength[0] <= wave <= self.wavelength[-1]:
            return True
        else:
            return False

    def __call__(self, index: int, wave: np.ndarray) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return self.interp[index](wave)
