# EEMs-toolkit — API Reference

> [中文文档](API.zh-CN.md) ｜ Auto-generated from source; covers every public module, class, function and constant under `src/EEMs_toolkit/`.

Auto-generated reference of every public module, class, function and constant.

## Contents

- Quick start
- Core library (scripting API)
  - [EEMs_toolkit](#eems-toolkit)
  - [EEMs_toolkit.constant](#eems-toolkit-constant)
  - [EEMs_toolkit.loader](#eems-toolkit-loader)
  - [EEMs_toolkit.absorbance](#eems-toolkit-absorbance)
  - [EEMs_toolkit.eem](#eems-toolkit-eem)
  - [EEMs_toolkit.parafac](#eems-toolkit-parafac)
  - [EEMs_toolkit.export](#eems-toolkit-export)
  - [EEMs_toolkit.project](#eems-toolkit-project)
- Web subpackage (FastAPI endpoints)
  - [EEMs_toolkit.web](#eems-toolkit-web)
  - [EEMs_toolkit.web.app](#eems-toolkit-web-app)
  - [EEMs_toolkit.web.\_\_main__](#eems-toolkit-web-main)
  - [EEMs_toolkit.web.session](#eems-toolkit-web-session)
  - [EEMs_toolkit.web.jobs](#eems-toolkit-web-jobs)
  - [EEMs_toolkit.web.payloads](#eems-toolkit-web-payloads)
  - [EEMs_toolkit.web.schemas](#eems-toolkit-web-schemas)
  - [EEMs_toolkit.web.routers.data](#eems-toolkit-web-routers-data)
  - [EEMs_toolkit.web.routers.preprocess](#eems-toolkit-web-routers-preprocess)
  - [EEMs_toolkit.web.routers.parafac](#eems-toolkit-web-routers-parafac)
  - [EEMs_toolkit.web.routers.analysis](#eems-toolkit-web-routers-analysis)

### Quick start

```python
import EEMs_toolkit as et

# 1. Read data
eems = et.read_eems(file_list=['s1.xlsx', 's2.xlsx'], eem_path='eem/')
abs_data = et.read_abs(file_list=['s1.xlsx'], abs_path='abs/')
blank = et.read_blank(file_list=['blank.xlsx'], blank_path='blank/')

# 2. Preprocess
eems.inner_effect_correct(abs_data)      # inner filter effect correction
eems.minus_the_blank(blank)               # subtract blank
eems.raman_areal(blank)                   # Raman normalisation AU → RU
eems.cut_ray_scatter((10, 10), (10, 10))  # remove Rayleigh scatter
eems.cut_ram_scatter((10, 10), (10, 10))  # remove Raman scatter
eems.miss_value_interpolation()           # interpolate missing values

# 3. PARAFAC (fit 2 and 3 components together)
eems.multi_non_parafac_cal(f=[2, 3], starts=10, max_iter=2500)
eems.plot_core_consistency_and_explanation()
eems.plot_sse()                           # compare SSE across component numbers
eems.plot_outlier_test(f=3)

# 4. Split-half validation
eems.split_analysis(f=3, starts=10)

# 5. Indices / FRI / spectral slope
eems.fluorescence_indices_output()
eems.fri_result_output()
abs_data.slope_fit()

# 6. Export
eems.open_fluor(f=3)
eems.parafac_result_output(f=3)
```

See the Web HTTP API section for the FastAPI endpoints.


---

## Core library (scripting API)

## EEMs_toolkit

<a id="eems-toolkit"></a>

**Module**: `EEMs_toolkit`

### Overview

Top-level package. Re-exports the public scripting API: EEMsDataset, Abs, read_eems, read_abs and read_blank.


## EEMs_toolkit.constant

<a id="eems-toolkit-constant"></a>

**Module**: `EEMs_toolkit.constant`

### Overview

Physical constants and wavelength regions shared across the toolkit (peaks, indices, Raman, split-half).

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `BIX` | tuple | `((380, 430), 310)` | Biological index region: Em 380 vs 430 at Ex 310. |
| `CROSS` | tuple | `(((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))` | The three cross-validation pair combinations for the 4 splits. |
| `FLUI` | tuple | `((470, 520), 370)` | Fluorescence index region: Em 470–520 at Ex 370. |
| `FREI` | tuple | `((380, (420, 430)), 310)` | Freshness index region: Em 380 / (420–430) at Ex 310. |
| `HIX` | tuple | `(((435, 480), (300, 345)), 254)` | Humification index region: Em 435–480 vs 300–345 at Ex 254. |
| `LONG_RANGE` | tuple | `(300, 600)` | Default long-wavelength range (300, 600) nm used by slope fitting. |
| `N_SPLIT` | int | `4` | Number of sample splits (4) used by the S4C6T3 split-half analysis. |
| `PEAKS` | str | `'BTAMCDN'` | String of the seven classic EEM peak identifiers: B, T, A, M, C, D, N. |
| `PEAK_A` | tuple | `((400, 460), 260)` | Peak A region (Em 400–460, Ex 260). |
| `PEAK_B` | tuple | `(305, 275)` | Peak B position (Em 305, Ex 275). |
| `PEAK_C` | tuple | `((420, 460), (320, 360))` | Peak C region (Em 420–460, Ex 320–360). |
| `PEAK_D` | tuple | `(509, 390)` | Peak D position (Em 509, Ex 390). |
| `PEAK_M` | tuple | `((370, 410), (290, 310))` | Peak M region (Em 370–410, Ex 290–310). |
| `PEAK_N` | tuple | `(370, 280)` | Peak N position (Em 370, Ex 280). |
| `PEAK_T` | tuple | `(340, 275)` | Peak T position (Em 340, Ex 275). |
| `RAMAN_EM_RANGE` | tuple | `(378, 424)` | Default emission range (378, 424) nm used to integrate the Raman peak. |
| `RAMAN_EX` | int | `350` | Default excitation wavelength (350 nm) used for Raman normalisation. |
| `S_275_295` | tuple | `(275, 295)` | Wavelength range (275, 295) nm for the short-wavelength spectral slope S. |
| `S_350_400` | tuple | `(350, 400)` | Wavelength range (350, 400) nm for the long-wavelength spectral slope S. |
| `WATER_FREQ` | int | `3382` | Raman shift of water (3382 cm⁻¹) used to locate Raman scatter. |

## EEMs_toolkit.loader

<a id="eems-toolkit-loader"></a>

**Module**: `EEMs_toolkit.loader`

### Overview

File readers for EEM, absorbance and blank samples.

#### Functions

- **`read_abs`**
  - `read_abs(file_list: 'Iterable[str]', abs_path: 'str') -> 'Abs'`
  - Read the Abs samples.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `file_list` | A list of abs filenames. |
  | `abs_path` | The directory where the Abs files saved. |
  - **Returns**: Absorbance dataset.

- **`read_blank`**
  - `read_blank(file_list: 'Iterable[str]', blank_path: 'str') -> 'np.ndarray[tuple[int, int, int], np.dtype[np.float64]]'`
  - Read the blank EEM samples with different Ex in columns and different Em in rows. The header is Ex and the index is Em.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `file_list` | A list of blank filenames. |
  | `blank_path` | The directory where the blank EEMs files saved. |
  - **Returns**: blank_eem

- **`read_eems`**
  - `read_eems(file_list: 'Iterable[str]', eem_path: 'str') -> 'EEMsDataset'`
  - Read the EEM samples with different Ex in columns and different Em in rows. The header is Ex and the index is Em.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `file_list` | A list of eem filenames. |
  | `eem_path` | The directory where the EEM files saved. |
  - **Returns**: EEMsDataset.

## EEMs_toolkit.absorbance

<a id="eems-toolkit-absorbance"></a>

**Module**: `EEMs_toolkit.absorbance`

### Overview

Absorbance dataset container and spectral-slope fitting.

#### Classes

##### `Abs`
The object of absorbance dataset

###### Methods

- **`check_wavelength`**
  - `check_wavelength(self, wave: 'float') -> 'bool'`

- **`slope_fit`**
  - `slope_fit(self, long_range: 'tuple[int, int]' = (300, 600)) -> 'pd.DataFrame'`
  - Fit spectral slopes in two different areas of the CDOM absorbance spectrum. It will create a new excel file named 'slope_fit_result.xlsx' in the current directory. Exponential model: y = a350*exp(S/1000*(350-lambda))+k Reference DOI: 10.4319/lo.2008.53.3.0955, 10.4319/lo.2001.46.8.2087

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `long_range` | Wavelength range for the long wavelength slope. Default is (300, 600). |
  - **Returns**: A DataFrame of results.

## EEMs_toolkit.eem

<a id="eems-toolkit-eem"></a>

**Module**: `EEMs_toolkit.eem`

### Overview

The EEMsDataset class: preprocessing, PARAFAC modelling, indices/FRI and plotting.

#### Classes

##### `EEMsDataset`
The object of EEMs dataset.

###### Methods

- **`copy`**
  - `copy(self) -> "'EEMsDataset'"`
  - Return an independent copy of this dataset.

- **`cut_ram_scatter`**
  - `cut_ram_scatter(self, first: 'tuple[float, float]', second: 'tuple[float, float]', freq: 'float' = 3382)`
  - Remove Raman scatter in place, use nans instead.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `first` | A tuple with two elements, representing widths below and above 1st Raman scatter center. |
  | `second` | A tuple with two elements, representing widths below and above 2nd Raman scatter center. |
  | `freq` | Optional, a constant named Raman shift which is around 3400-3600 cm^-1 for water. Default is 3382. |
- **`cut_ray_scatter`**
  - `cut_ray_scatter(self, first: 'tuple[float, float]', second: 'tuple[float, float]')`
  - Remove Rayleigh scatter in place, use nans instead.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `first` | A tuple with two elements, representing widths below and above 1st Rayleigh scatter center. |
  | `second` | A tuple with two elements, representing widths below and above 2nd Rayleigh scatter center. |
- **`dilute`**
  - `dilute(self, dilution_factors: 'Iterable[float]')`
  - Dilution correction in place.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `dilution_factors` | A list of dilution factors for every sample. |
- **`eems_output`**
  - `eems_output(self, output_path: 'Optional[str]' = None)`
  - Output the EEMs

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `output_path` | Optional, the directory to save the text. If not given, save in the current directory. |
- **`fluorescence_indices_output`**
  - `fluorescence_indices_output(self, output_path: 'Optional[str]' = None, output_name: 'str' = 'fluorescence_indices.xlsx')`
  - Calculate fluorescence indices and output

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `output_path` | Optional, the directory to save the text. If not given, save in the current directory. |
- **`fri_result_output`**
  - `fri_result_output(self, output_path: 'Optional[str]' = None, filename: 'str' = 'FRI.xlsx')`
  - Calculate FRI and output

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `output_path` | Optional, the directory to save the text. If not given, save in the current directory. |
- **`inner_effect_correct`**
  - `inner_effect_correct(self, absorbance: 'Abs')`
  - Inner filter effect correction in place. Reference DOI: 10.1021/es0155276, 10.4319/lom.2013.11.616

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `absorbance` | An Abs object. |
- **`minus_the_blank`**
  - `minus_the_blank(self, blank: 'np.ndarray[tuple[int, int] | tuple[int, int, int], np.dtype[np.float64]]')`
  - Minus blank sample in place.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `blank` | The same shape as self.x or just one sample. |
- **`miss_value_interpolation`**
  - `miss_value_interpolation(self)`
  - Use Delaunay triangulation method to interpolate in place, aim at removing nans in EEMs so that the parafac program can run normally. Reference DOI: 10.1016/j.marchem.2004.02.006

- **`multi_non_parafac_cal`**
  - `multi_non_parafac_cal(self, f: 'Optional[int | Iterable[int]]' = None, tol: 'float' = 1e-06, max_iter: 'int' = 2500, starts: 'int' = 10, *, n_processes: 'Optional[int]' = None, progress=None, stop_event=None)`
  - Run 'start' times Non-negative CP decomposition via HALS. Uses Hierarchical ALS (Alternating Least Squares) which updates each factor column-wise (one column at a time while keeping all other columns fixed)

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, Iterable, numbers of components |
  | `tol` | float, optional, the algorithm stops when the variation in the reconstruction error is less than the tol. Default: 1e-6 |
  | `max_iter` | int, maximum number of iteration |
  | `starts` | int, number of models to fit |
  | `n_processes` | Optional, number of worker processes, defaults to min(starts, 8). |
  | `progress` | Optional, callback called as progress(done, total, f). |
  | `stop_event` | Optional, threading.Event used to cancel the job. |
- **`open_fluor`**
  - `open_fluor(self, f: 'int', output_path: 'Optional[str]' = None, filename: 'str' = 'model.txt', **kwargs)`
  - Output a text file which can be uploaded to the OpenFluor to compare.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, number of components |
  | `output_path` | Optional, the directory to save the text. If not given, save in the current directory. |
  | `kwargs` | Optional, consist of personal information |
- **`parafac_result_output`**
  - `parafac_result_output(self, f: 'int', output_path: 'Optional[str]' = None, filename: 'str' = 'parafac-result.xlsx')`
  - Output PARAFAC results, including FMax, Em loadings, Ex loadings and components.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, number of components |
  | `output_path` | Optional, the directory to save the text. If not given, save in the current directory. |
- **`plot_core_consistency_and_explanation`**
  - `plot_core_consistency_and_explanation(self)`
  - Plot core consistency and explanation rate, with f in self.f

- **`plot_eem_by1`**
  - `plot_eem_by1(self, f: 'Optional[int]' = None, sample_id: 'Optional[np.typing.ArrayLike]' = None, plot_3d: 'bool' = False)`
  - Plot EEMs

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | Optional, if given, plot model from factors instead self.x. Default is None. |
  | `sample_id` | Optional, which samples to plot, if not given, plot all. Default is None. |
  | `plot_3d` | Optional, if True then plot 3D else contourf. Default is False. |
- **`plot_fingers`**
  - `plot_fingers(self, f: 'int')`
  - Plot components in one figure. It will print peaks of each component for it's Ex and Em.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, number of components |
- **`plot_fmax`**
  - `plot_fmax(self, f: 'int', use_file_list: 'bool' = False)`
  - Plot FMax to see components' variation among samples.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, number of components |
  | `use_file_list` | Optional, if True, use file_list in xticks. Default is False |
- **`plot_fri`**
  - `plot_fri(self, sample_id: 'Optional[Sequence[int]]' = None)`
  - Plot FRI Reference DOI: 10.1021/es034354c

- **`plot_loadings`**
  - `plot_loadings(self, f: 'int')`
  - Plot loadings of components

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, number of components |
- **`plot_outlier_test`**
  - `plot_outlier_test(self, f: 'int')`
  - Create a new figure to show outliers.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, number of components |
- **`plot_residual_error`**
  - `plot_residual_error(self, f: 'int', sample_id: 'Optional[np.typing.ArrayLike]' = None)`
  - Plot sample, model from factors and residual in one figure.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, number of component |
  | `sample_id` | Optional, which samples to plot, if not given, plot all. Default is None. |
- **`plot_split_result`**
  - `plot_split_result(self, f: 'int')`
  - Plot split analysis result

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, number of components |
- **`plot_sse`**
  - `plot_sse(self, f: 'Optional[int | Iterable[int]]' = None)`
  - Plot SSE for one or more component numbers. The first panel shows the total SSE against the number of components, the other three show the SSE aggregated per sample, per emission and per excitation wavelength so that different component numbers can be compared.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, Iterable, numbers of components. Defaults to the fitted models. |
- **`raman_areal`**
  - `raman_areal(self, blank: 'np.ndarray[tuple[int, int] | tuple[int, int, int], np.dtype[np.float64]]', ex: 'float' = 350, em_range: 'tuple[float, float]' = (378, 424))`
  - Turning arbitrary units(AU) into Raman units(RU). Reference DOI: 10.1366/000370209788964548

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `blank` | The same shape as self.x or just one sample. |
  | `ex` | Optional, the excitation position of Raman peak. Default is 350. |
  | `em_range` | Optional, a tuple of two elements, range used to calculate the area, if not given, it will be (378, 424). |
- **`restore`**
  - `restore(self, snapshot: 'dict')`
  - Restore the mutable state in place from :meth:`snapshot` output.

- **`smooth_eem`**
  - `smooth_eem(self, sigma: 'float' = 0.5, filter_size: 'Optional[int]' = None)`
  - Use Gaussian kernel to smooth EEMs in place. Suggesting after interpolation.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `sigma` | Optional, standard deviation of the Gaussian distribution, specified as a positive number. Default is 0.5. |
  | `filter_size` | optional, size of the Gaussian filter, specified as a positive, odd integer. The default filter size is 2\*ceil(2\*sigma)+1. |
- **`snapshot`**
  - `snapshot(self) -> 'dict'`
  - Return a deep copy of the mutable state (used by the history timeline).

- **`split_analysis`**
  - `split_analysis(self, f: 'Optional[int | Iterable[int]]' = None, tol: 'float' = 1e-06, max_iter: 'int' = 2500, starts: 'int' = 10, random_state: 'Optional[int]' = None, *, n_processes: 'Optional[int]' = None, progress=None, stop_event=None)`
  - Generate PARAFAC models nested in dataset splits. TCC above 0.95 can be validated. Reference DOI: 10.1027/1614-2241.2.2.57

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, number of components |
  | `random_state` | {None(default), int, np.random.RandomState} |
  | `starts` | int, number of models to fit |
  | `n_processes` | Optional, number of worker processes, defaults to min(starts, 8). |
  | `progress` | Optional, callback called as progress(done, total, f). |
  | `stop_event` | Optional, threading.Event used to cancel the job. |
- **`split_validation`**
  - `split_validation(self, f: 'Optional[int | Iterable[int]]' = None)`

- **`sse_analysis`**
  - `sse_analysis(self, f: 'Optional[int | Iterable[int]]' = None) -> 'dict[int, dict]'`
  - Compute the sum of squared errors (SSE) for fitted models. The squared error tensor is aggregated along each of the three axes: per sample, per emission wavelength and per excitation wavelength.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `f` | int, Iterable, numbers of components. Defaults to the fitted models. |
  - **Returns**: {f: {'total': float, 'sample': ndarray, 'em': ndarray, 'ex': ndarray}}

- **`sub_dataset`**
  - `sub_dataset(self, sample_orders: 'Optional[list[int]]' = None, ex_orders: 'Optional[np.ndarray[tuple[int], np.dtype[np.bool_]]]' = None, em_orders: 'Optional[np.ndarray[tuple[int], np.dtype[np.bool_]]]' = None)`
  - Removing samples and/or wavelengths in place.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `sample_orders` | A list of positive integers, 1 is the first sample. |
  | `ex_orders` | e.g. self.ex == 240 |
  | `em_orders` | e.g. self.em > 600 |
## EEMs_toolkit.parafac

<a id="eems-toolkit-parafac"></a>

**Module**: `EEMs_toolkit.parafac`

### Overview

Low-level PARAFAC engine: non-negative CP (HALS), split-half fitting and Tucker congruence validation.

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `ProgressCallback` | _CallableGenericAlias | `typing.Callable[[int, int, int], NoneType]` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Classes

##### `FitResult`
Result of fitting models with several component numbers.

###### Fields

| Name | Type | Value |
|---|---|---|
| `factors` | `dict[int, CPTensor]` | `{}` |
| `leverage` | `dict[int, np.ndarray]` | `{}` |
| `sse` | `dict[int, np.ndarray]` | `{}` |
| `core_consistency` | `dict[int, float]` | `{}` |
| `explanation_rate` | `dict[int, float]` | `{}` |
| `log` | `list[str]` | `[]` |

##### `JobCancelled`
Raised when a long-running fit is stopped by the caller.
- Bases: `Exception`

##### `SplitResult`
Result of split-half analysis.

###### Fields

| Name | Type | Value |
|---|---|---|
| `split_result` | `dict[int, list[CPTensor]]` | `{}` |
| `splits_index` | `list[np.ndarray]` | `[]` |
| `log` | `list[str]` | `[]` |
#### Functions

- **`component_numbers`**
  - `component_numbers(f: 'Optional[int | Iterable[int]]', default: 'Iterable[int]' = range(2, 6)) -> 'tuple[int, ...]'`
  - Normalize the requested component numbers into a tuple.

- **`fit_models`**
  - `fit_models(x: 'np.ndarray', fs: 'Iterable[int]', *, starts: 'int' = 10, tol: 'float' = 1e-06, max_iter: 'int' = 2500, n_processes: 'Optional[int]' = None, progress: 'Optional[ProgressCallback]' = None, stop_event=None) -> 'FitResult'`
  - Fit several starts for every requested number of components and keep the best model.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `x` | EEM data with shape (nSample, nem, nex). |
  | `fs` | Numbers of components to fit. |
  | `starts` | Number of random initializations per number of components. |
  | `tol` | Convergence tolerance of the HALS algorithm. |
  | `max_iter` | Maximum number of iterations of one fit. |
  | `n_processes` | Worker processes, defaults to min(starts, 8). |
  | `progress` | Optional callback called as progress(done, total, f). |
  | `stop_event` | Optional threading.Event used to cancel the job. |
- **`process_count`**
  - `process_count(starts: 'int', n_processes: 'Optional[int]' = None) -> 'int'`
  - Number of worker processes, capped at 8 when starts is larger than 8.

- **`split_fit`**
  - `split_fit(x: 'np.ndarray', factors: 'dict[int, CPTensor]', fs: 'Iterable[int]', *, starts: 'int' = 10, tol: 'float' = 1e-06, max_iter: 'int' = 2500, random_state: 'Optional[int]' = None, n_processes: 'Optional[int]' = None, progress: 'Optional[ProgressCallback]' = None, stop_event=None) -> 'SplitResult'`
  - Generate PARAFAC models nested in dataset splits (S4C6T3).

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `x` | EEM data with shape (nSample, nem, nex). |
  | `factors` | Overall models, used to align the order of components. |
  | `fs` | Numbers of components to fit. |
  | `random_state` | Optional seed of the sample shuffling. |
- **`split_validation`**
  - `split_validation(factors: 'dict[int, CPTensor]', split_result: 'dict[int, list[CPTensor]]', fs: 'Iterable[int]') -> 'dict[int, dict]'`
  - Compare split models with the overall model using Tucker congruence coefficients.

  - **Returns**: Mapping of component number to {min_tcc, validated, em_tcc, ex_tcc}.

## EEMs_toolkit.export

<a id="eems-toolkit-export"></a>

**Module**: `EEMs_toolkit.export`

### Overview

Builders that turn dataset results into bytes, shared by the file writers and the web API.

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `DEFAULT_EXPORTS` | tuple | `('openfluor', 'parafac', 'eems', 'indices', 'fri', 'slope')` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Functions

- **`all_zip_bytes`**
  - `all_zip_bytes(dataset, abs_obj=None, f: 'Optional[int]' = None, include: 'Optional[Iterable[str]]' = None, **kwargs) -> 'bytes'`
  - Build a zip archive with every requested export. Unavailable items are listed in ``skipped.txt`` instead of failing the whole archive.

- **`eem_xlsx_bytes`**
  - `eem_xlsx_bytes(dataset, index: 'int') -> 'bytes'`
  - Build one EEM workbook with Ex on the first row and Em on the first column.

- **`eems_zip_bytes`**
  - `eems_zip_bytes(dataset) -> 'bytes'`
  - Build a zip archive of all EEM workbooks.

- **`fri_xlsx_bytes`**
  - `fri_xlsx_bytes(dataset) -> 'bytes'`
  - Build the FRI workbook. Raises ValueError when the wavelength ranges are not covered.

- **`indices_xlsx_bytes`**
  - `indices_xlsx_bytes(dataset) -> 'bytes'`
  - Build the fluorescence indices workbook.

- **`open_fluor_text`**
  - `open_fluor_text(dataset, f: 'int', **kwargs) -> 'bytes'`
  - Build the OpenFluor text file content.

- **`parafac_xlsx_bytes`**
  - `parafac_xlsx_bytes(dataset, f: 'int') -> 'bytes'`
  - Build the PARAFAC result workbook (FMax, loadings and component maps).

- **`slope_xlsx_bytes`**
  - `slope_xlsx_bytes(abs_obj) -> 'bytes'`
  - Build the absorbance slope workbook.

## EEMs_toolkit.project

<a id="eems-toolkit-project"></a>

**Module**: `EEMs_toolkit.project`

### Overview

Snapshot helpers and the ``.eemproj`` project archive format.

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `FORMAT_VERSION` | int | `1` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Classes

##### `LoadedProject`
Everything recovered from a ``.eemproj`` archive.

###### Fields

| Name | Type | Value |
|---|---|---|
| `dataset` | `Any` | `` |
| `history` | `list[dict]` | `[]` |
| `log` | `list[dict]` | `[]` |
| `selected_f` | `Optional[int]` | `None` |
| `sources` | `dict` | `{}` |
| `dilution` | `Optional[np.ndarray]` | `None` |
| `abs_obj` | `Any` | `None` |
| `blank` | `Optional[np.ndarray]` | `None` |
| `pipeline` | `list[dict]` | `[]` |
| `history_index` | `int` | `0` |
#### Functions

- **`apply_snapshot`**
  - `apply_snapshot(dataset, snapshot: 'dict') -> 'None'`
  - Restore a dataset in place from :func:`take_snapshot` output.

- **`dump_project`**
  - `dump_project(fileobj, dataset, *, history: 'Optional[list[dict]]' = None, log: 'Optional[list[dict]]' = None, selected_f: 'Optional[int]' = None, sources: 'Optional[dict]' = None, dilution: 'Optional[np.ndarray]' = None, abs_obj=None, blank: 'Optional[np.ndarray]' = None, pipeline: 'Optional[list[dict]]' = None, history_index: 'int' = 0, include_history: 'bool' = True) -> 'None'`
  - Write a ``.eemproj`` archive.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `fileobj` | A path or a binary file object. |
  | `history` | Step metadata; each entry may hold ``path`` pointing to a snapshot npz. |
  | `log` | Pipeline log entries shown in the UI. |
- **`load_project`**
  - `load_project(fileobj, *, history_dir: 'Optional[str]' = None) -> 'LoadedProject'`
  - Read a ``.eemproj`` archive.

  **Parameters**

  | Parameter | Description |
  |---|---|
  | `fileobj` | A path or a binary file object. |
  | `history_dir` | Optional directory receiving the extracted history snapshots. |
- **`load_snapshot`**
  - `load_snapshot(fileobj) -> 'dict'`
  - Read a snapshot written by :func:`save_snapshot`.

- **`rebuild_cp_tensor`**
  - `rebuild_cp_tensor(weights: 'np.ndarray', sample: 'np.ndarray', em: 'np.ndarray', ex: 'np.ndarray') -> 'CPTensor'`
  - Build a CPTensor with the factor order used by this package: sample, em, ex.

- **`save_snapshot`**
  - `save_snapshot(fileobj, snapshot: 'dict') -> 'None'`
  - Save a snapshot into an npz archive (path or binary file object).

- **`take_snapshot`**
  - `take_snapshot(dataset) -> 'dict'`
  - Deep copy the mutable state of a dataset.

---

## Web subpackage (FastAPI endpoints)

## EEMs_toolkit.web

<a id="eems-toolkit-web"></a>

**Module**: `EEMs_toolkit.web`

### Overview

Optional local web interface for EEMs-toolkit.


## EEMs_toolkit.web.app

<a id="eems-toolkit-web-app"></a>

**Module**: `EEMs_toolkit.web.app`

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `COOKIE_NAME` | str | `'eems_sid'` | Internal constant of the web subpackage; not part of the public scripting API. |
| `STATIC_DIR` | WindowsPath | `WindowsPath('EEMs_toolkit/web/static')` | Internal constant of the web subpackage; not part of the public scripting API. |
| `app` | FastAPI | `` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Functions

- **`create_app`**
  - `create_app() -> 'FastAPI'`

- **`lifespan`**
  - `lifespan(_: 'FastAPI') -> 'AsyncGenerator[None]'`

## EEMs_toolkit.web.\_\_main\_\_

<a id="eems-toolkit-web-main"></a>

**Module**: `EEMs_toolkit.web.__main__`

### Overview

``eems-web`` entry point: start uvicorn and open the browser.

#### Functions

- **`main`**
  - `main(argv: 'list[str] | None' = None) -> 'None'`

## EEMs_toolkit.web.session

<a id="eems-toolkit-web-session"></a>

**Module**: `EEMs_toolkit.web.session`

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `MAX_HISTORY` | int | `50` | Internal constant of the web subpackage; not part of the public scripting API. |
| `MAX_SESSIONS` | int | `20` | Internal constant of the web subpackage; not part of the public scripting API. |
| `SESSION_TTL` | int | `14400` | Internal constant of the web subpackage; not part of the public scripting API. |
| `STORE` | SessionStore | `` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Classes

##### `HistoryLimitReached`
Raised when the timeline already holds MAX_HISTORY steps.
- Bases: `Exception`

##### `Session`
State of one browser session: dataset, timeline and imported sources.

###### Methods

- **`close`**
  - `close(self) -> 'None'`

- **`invalidate_models`**
  - `invalidate_models(self) -> 'None'`

- **`jump`**
  - `jump(self, index: 'int') -> 'None'`

- **`previous_snapshot`**
  - `previous_snapshot(self) -> 'Optional[dict]'`
  - Snapshot of the step before the current position, for before/after previews.

- **`record`**
  - `record(self, action: 'str', params: 'Optional[dict]' = None) -> 'None'`
  - Append the current dataset state to the timeline after a successful step.

- **`replace_from_project`**
  - `replace_from_project(self, loaded) -> 'None'`
  - Adopt a LoadedProject without copying its data again.

- **`set_dataset`**
  - `set_dataset(self, dataset, *, sources=None, abs_obj=None, blank=None, dilution=None) -> 'None'`

- **`touch`**
  - `touch(self) -> 'None'`

- **`trim`**
  - `trim(self, index: 'int') -> 'None'`
  - Delete the step at ``index`` and every later step.

##### `SessionStore`
Bounded, TTL-based session registry.

###### Methods

- **`cleanup`**
  - `cleanup(self) -> 'None'`

- **`close_all`**
  - `close_all(self) -> 'None'`

- **`get`**
  - `get(self, sid: 'Optional[str]') -> 'Optional[Session]'`

- **`get_or_create`**
  - `get_or_create(self, sid: 'Optional[str]' = None) -> 'Session'`
#### Functions

- **`get_session`**
  - `get_session(request: 'Request') -> 'Session'`
  - FastAPI dependency returning the session bound by the middleware.

## EEMs_toolkit.web.jobs

<a id="eems-toolkit-web-jobs"></a>

**Module**: `EEMs_toolkit.web.jobs`

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `MANAGER` | JobManager | `` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Classes

##### `Job`
Job(id: 'str', kind: 'str', session_id: 'str', state: 'str' = 'running', done: 'int' = 0, total: 'int' = 0, phase: 'str' = '', log: 'list[str]' = <factory>, error: 'Optional[str]' = None, started: 'float' = <factory>, finished: 'Optional[float]' = None, stop_event: 'threading.Event' = <factory>, revision: 'int' = 0)

###### Fields

| Name | Type | Value |
|---|---|---|
| `id` | `str` | `` |
| `kind` | `str` | `` |
| `session_id` | `str` | `` |
| `state` | `str` | `` |
| `done` | `int` | `` |
| `total` | `int` | `` |
| `phase` | `str` | `` |
| `log` | `list[str]` | `` |
| `error` | `Optional[str]` | `` |
| `started` | `float` | `` |
| `finished` | `Optional[float]` | `` |
| `stop_event` | `threading.Event` | `` |
| `revision` | `int` | `` |

###### Methods

- **`add_log`**
  - `add_log(self, line: 'str') -> 'None'`

- **`progress`**
  - `progress(self, done: 'int', total: 'int', f: 'int') -> 'None'`

- **`snapshot`**
  - `snapshot(self) -> 'dict'`

##### `JobManager`
One long-running job per session at a time.

###### Methods

- **`active`**
  - `active(self, session_id: 'str') -> 'Optional[Job]'`

- **`cleanup`**
  - `cleanup(self) -> 'None'`

- **`create`**
  - `create(self, session_id: 'str', kind: 'str') -> 'Job'`

- **`get`**
  - `get(self, job_id: 'str') -> 'Optional[Job]'`

- **`stop`**
  - `stop(self, job: 'Job') -> 'None'`

## EEMs_toolkit.web.payloads

<a id="eems-toolkit-web-payloads"></a>

**Module**: `EEMs_toolkit.web.payloads`

### Overview

Turn dataset arrays into plain JSON payloads for the Plotly charts.

#### Functions

- **`core_payload`**
  - `core_payload(session) -> 'dict'`

- **`eem_payload`**
  - `eem_payload(session, sample: 'int', state: 'str' = 'current', model_f: 'Optional[int]' = None, stride: 'int' = 1) -> 'dict'`

- **`fingers_payload`**
  - `fingers_payload(session, f: 'int') -> 'dict'`

- **`fmax_payload`**
  - `fmax_payload(session, f: 'int') -> 'dict'`

- **`fri_payload`**
  - `fri_payload(session, sample: 'int') -> 'dict'`

- **`fri_table_payload`**
  - `fri_table_payload(session) -> 'dict'`

- **`history_payload`**
  - `history_payload(session) -> 'dict'`

- **`indices_payload`**
  - `indices_payload(session) -> 'dict'`

- **`loadings_payload`**
  - `loadings_payload(session, f: 'int') -> 'dict'`

- **`outlier_payload`**
  - `outlier_payload(session, f: 'int') -> 'dict'`

- **`residual_payload`**
  - `residual_payload(session, f: 'int', sample: 'int') -> 'dict'`

- **`samples_payload`**
  - `samples_payload(session) -> 'list[dict]'`

- **`slope_payload`**
  - `slope_payload(session) -> 'dict'`

- **`split_payload`**
  - `split_payload(session, f: 'int') -> 'dict'`

- **`sse_payload`**
  - `sse_payload(session) -> 'dict'`

- **`state_payload`**
  - `state_payload(session) -> 'dict'`

## EEMs_toolkit.web.schemas

<a id="eems-toolkit-web-schemas"></a>

**Module**: `EEMs_toolkit.web.schemas`

#### Classes

##### `CutScatterRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `ray` | `tuple[float, float]` | `` |
| `second_ray` | `tuple[float, float]` | `` |
| `ram` | `tuple[float, float]` | `` |
| `second_ram` | `tuple[float, float]` | `` |

##### `DiluteRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `factors` | `typing.Optional[list[float]]` | `` |

##### `ExcludeSamplesRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `orders` | `list[int]` | `` |

##### `ExportAllRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `include` | `typing.Optional[list[str]]` | `` |
| `f` | `typing.Optional[int]` | `` |

##### `HistoryJumpRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `index` | `int` | `` |

##### `ParafacJobRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `f` | `list[int]` | `` |
| `starts` | `int` | `10` |
| `tol` | `float` | `1e-06` |
| `max_iter` | `int` | `2500` |
| `n_processes` | `typing.Optional[int]` | `` |

##### `ProjectSaveRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `include_history` | `bool` | `True` |
| `f` | `typing.Optional[int]` | `` |

##### `RamanRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `ex` | `<class 'float'>` | `350` |
| `em_range` | `tuple[float, float]` | `(378, 424)` |

##### `SmoothRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `sigma` | `float` | `0.5` |
| `filter_size` | `typing.Optional[int]` | `` |

##### `SplitJobRequest`
- Bases: `ParafacJobRequest`

###### Fields

| Name | Type | Value |
|---|---|---|
| `f` | `list[int]` | `` |
| `starts` | `int` | `10` |
| `tol` | `float` | `1e-06` |
| `max_iter` | `int` | `2500` |
| `n_processes` | `typing.Optional[int]` | `` |
| `random_state` | `typing.Optional[int]` | `` |

##### `SubDatasetRequest`
- Bases: `BaseModel`

###### Fields

| Name | Type | Value |
|---|---|---|
| `sample_orders` | `typing.Optional[list[int]]` | `` |
| `ex_lt` | `typing.Optional[float]` | `` |
| `ex_gt` | `typing.Optional[float]` | `` |
| `em_lt` | `typing.Optional[float]` | `` |
| `em_gt` | `typing.Optional[float]` | `` |

## EEMs_toolkit.web.routers.data

<a id="eems-toolkit-web-routers-data"></a>

**Module**: `EEMs_toolkit.web.routers.data`

### Overview

Data import, session state, history timeline and project archives.

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `DEMO_PACKAGE` | WindowsPath | `WindowsPath('EEMs_toolkit/web/demo/eems-demo.zip')` | Internal constant of the web subpackage; not part of the public scripting API. |
| `router` | APIRouter | `` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Functions

- **`demo_package`**
  - `demo_package()`
  - Download the demo archive (EEM / abs / blank / SampleLog) used in the documentation.

- **`get_history`**
  - `get_history(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`get_samples`**
  - `get_samples(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`get_state`**
  - `get_state(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`import_files`**
  - `import_files(eem_files: 'list[UploadFile]' = File([]), abs_files: 'list[UploadFile]' = File([]), blank_files: 'list[UploadFile]' = File([]), sample_log: 'Optional[UploadFile]' = File(None), zip_file: 'Optional[UploadFile]' = File(None), session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`jump_history`**
  - `jump_history(request: 'HistoryJumpRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`load_project`**
  - `load_project(file: 'UploadFile' = File(PydanticUndefined), session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`save_project`**
  - `save_project(request: 'ProjectSaveRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`trim_history`**
  - `trim_history(request: 'HistoryJumpRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`
#### HTTP Routes

| Method | Path | Handler | Description |
|---|---|---|---|
| GET | `/api/state` | `get_state` |  |
| GET | `/api/samples` | `get_samples` |  |
| POST | `/api/import` | `import_files` |  |
| GET | `/api/import/demo-package` | `demo_package` | Download the demo archive (EEM / abs / blank / SampleLog) used in the documentation. |
| GET | `/api/history` | `get_history` |  |
| POST | `/api/history/jump` | `jump_history` |  |
| POST | `/api/history/trim` | `trim_history` |  |
| POST | `/api/project/save` | `save_project` |  |
| POST | `/api/project/load` | `load_project` |  |

## EEMs_toolkit.web.routers.preprocess

<a id="eems-toolkit-web-routers-preprocess"></a>

**Module**: `EEMs_toolkit.web.routers.preprocess`

### Overview

Preprocessing endpoints: every call appends a step to the session timeline.

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `router` | APIRouter | `` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Functions

- **`cut_scatter`**
  - `cut_scatter(request: 'CutScatterRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`dilute`**
  - `dilute(request: 'DiluteRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`inner_effect_correct`**
  - `inner_effect_correct(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`interpolate`**
  - `interpolate(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`minus_blank`**
  - `minus_blank(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`raman_areal`**
  - `raman_areal(request: 'RamanRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`smooth`**
  - `smooth(request: 'SmoothRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`sub_dataset`**
  - `sub_dataset(request: 'SubDatasetRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`
#### HTTP Routes

| Method | Path | Handler | Description |
|---|---|---|---|
| POST | `/api/preprocess/inner_effect_correct` | `inner_effect_correct` |  |
| POST | `/api/preprocess/minus_blank` | `minus_blank` |  |
| POST | `/api/preprocess/raman_areal` | `raman_areal` |  |
| POST | `/api/preprocess/sub_dataset` | `sub_dataset` |  |
| POST | `/api/preprocess/dilute` | `dilute` |  |
| POST | `/api/preprocess/smooth` | `smooth` |  |
| POST | `/api/preprocess/cut_scatter` | `cut_scatter` |  |
| POST | `/api/preprocess/interpolate` | `interpolate` |  |

## EEMs_toolkit.web.routers.parafac

<a id="eems-toolkit-web-routers-parafac"></a>

**Module**: `EEMs_toolkit.web.routers.parafac`

### Overview

PARAFAC / split-half jobs (start, stop, SSE) and chart payloads.

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `router` | APIRouter | `` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Functions

- **`chart_core`**
  - `chart_core(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`chart_eem`**
  - `chart_eem(sample: 'int' = Query(1), state: 'str' = 'current', model_f: 'Optional[int]' = None, stride: 'int' = Query(1), session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`chart_fingers`**
  - `chart_fingers(f: 'int', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`chart_fmax`**
  - `chart_fmax(f: 'int', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`chart_fri`**
  - `chart_fri(sample: 'int' = Query(1), session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`chart_loadings`**
  - `chart_loadings(f: 'int', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`chart_outlier`**
  - `chart_outlier(f: 'int', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`chart_residual`**
  - `chart_residual(f: 'int', sample: 'int' = Query(1), session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`chart_split`**
  - `chart_split(f: 'int', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`chart_sse`**
  - `chart_sse(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`get_job`**
  - `get_job(job_id: 'str')`

- **`job_events`**
  - `job_events(job_id: 'str')`

- **`list_jobs`**
  - `list_jobs(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`start_parafac`**
  - `start_parafac(request: 'ParafacJobRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`start_split`**
  - `start_split(request: 'SplitJobRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`stop_job`**
  - `stop_job(job_id: 'str')`
#### HTTP Routes

| Method | Path | Handler | Description |
|---|---|---|---|
| POST | `/api/jobs/parafac` | `start_parafac` |  |
| POST | `/api/jobs/split` | `start_split` |  |
| GET | `/api/jobs/{job_id}` | `get_job` |  |
| GET | `/api/jobs` | `list_jobs` |  |
| POST | `/api/jobs/{job_id}/stop` | `stop_job` |  |
| GET | `/api/jobs/{job_id}/events` | `job_events` |  |
| GET | `/api/charts/eem` | `chart_eem` |  |
| GET | `/api/charts/residual` | `chart_residual` |  |
| GET | `/api/charts/fingers` | `chart_fingers` |  |
| GET | `/api/charts/loadings` | `chart_loadings` |  |
| GET | `/api/charts/fmax` | `chart_fmax` |  |
| GET | `/api/charts/outlier` | `chart_outlier` |  |
| GET | `/api/charts/core` | `chart_core` |  |
| GET | `/api/charts/sse` | `chart_sse` |  |
| GET | `/api/charts/split` | `chart_split` |  |
| GET | `/api/charts/fri` | `chart_fri` |  |

## EEMs_toolkit.web.routers.analysis

<a id="eems-toolkit-web-routers-analysis"></a>

**Module**: `EEMs_toolkit.web.routers.analysis`

### Overview

Analysis panels (indices, FRI, slopes), outlier removal and the export centre.

#### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `XLSX_MEDIA` | str | `'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'` | Internal constant of the web subpackage; not part of the public scripting API. |
| `router` | APIRouter | `` | Internal constant of the web subpackage; not part of the public scripting API. |
#### Functions

- **`exclude_samples`**
  - `exclude_samples(request: 'ExcludeSamplesRequest', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`export_all`**
  - `export_all(f: 'int | None' = Query(None), include: 'str | None' = Query(None), session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`export_eems`**
  - `export_eems(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`export_fri`**
  - `export_fri(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`export_indices`**
  - `export_indices(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`export_openfluor`**
  - `export_openfluor(f: 'int', name: 'str' = '', creator: 'str' = '', email: 'str' = '', doi: 'str' = '', reference: 'str' = '', fluorometer: 'str' = '', sources: 'str' = '', ecozones: 'str' = '', method: 'str' = '', preprocess: 'str' = '', description: 'str' = '', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`export_parafac`**
  - `export_parafac(f: 'int', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`export_slope`**
  - `export_slope(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`get_fri`**
  - `get_fri(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`get_indices`**
  - `get_indices(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`get_outliers`**
  - `get_outliers(f: 'int', session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`

- **`get_slope`**
  - `get_slope(session: 'Session' = Depends(dependency=<function get_session >, use_cache=True, scope=None, max_concurrency=None))`
#### HTTP Routes

| Method | Path | Handler | Description |
|---|---|---|---|
| GET | `/api/indices` | `get_indices` |  |
| GET | `/api/fri` | `get_fri` |  |
| GET | `/api/slope` | `get_slope` |  |
| GET | `/api/outliers` | `get_outliers` |  |
| POST | `/api/samples/exclude` | `exclude_samples` |  |
| GET | `/api/export/openfluor` | `export_openfluor` |  |
| GET | `/api/export/parafac` | `export_parafac` |  |
| GET | `/api/export/eems` | `export_eems` |  |
| GET | `/api/export/indices` | `export_indices` |  |
| GET | `/api/export/fri` | `export_fri` |  |
| GET | `/api/export/slope` | `export_slope` |  |
| GET | `/api/export/all` | `export_all` |  |
