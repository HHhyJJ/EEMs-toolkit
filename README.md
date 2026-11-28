[English](README.md) | [中文文档](README.zh-CN.md)

# EEMs-toolkit - 0.3.0

EEMs-toolkit is a Python toolkit for analysing excitation–emission matrix (EEM) fluorescence data with
parallel factor analysis (PARAFAC). Its feature set is roughly comparable to the MATLAB drEEM toolbox.

It can be used in notebooks/scripts (matplotlib figures) or through an optional local web interface
(Plotly.js figures, Chinese/English UI).

API reference (auto-generated from source): [English](docs/API.md) / [中文](docs/API.zh-CN.md)

## Features

- **Data import**: EEM samples, absorbance spectra, blank samples and a sample log.
- **Preprocessing**: inner filter effect correction, blank subtraction, Raman normalisation (AU → RU),
  sample/wavelength cropping, dilution correction, Rayleigh/Raman scatter removal, missing-value
  interpolation (Delaunay), optional Gaussian smoothing.
- **PARAFAC**: non-negative models for one or more component numbers, multiple random starts,
  core consistency / explained variance, residuals, SSE analysis across component numbers
  (total and aggregated per sample / emission / excitation), leverage–SSE outlier diagnostics,
  sample exclusion, split-half analysis and validation (S4C6T3).
- **Results**: FMax, Ex/Em loadings, component “fingerprints”, OpenFluor text export and Excel export.
- **Other analyses**: fluorescence indices (FLUI, FREI, HIX, BIX), fluorescence regional integration
  (FRI, five regions), spectral slope / SR from absorbance spectra.
- **Projects**: save the whole session (data, preprocessing history, models) as a `.eemproj` file and
  reload it later.

## Installation

```bash
pip install EEMs-toolkit
```

## Optional web interface

```bash
pip install "EEMs-toolkit[web]"
eems-web                 # default http://127.0.0.1:8000
eems-web --port 8080     # custom port
eems-web --no-browser    # do not open a browser automatically
```

The web interface covers the full workflow: import → preprocessing (live preview, previous-step
comparison, restore to any history state) → PARAFAC (background jobs that can be stopped) →
split-half validation → indices / FRI / spectral slope → export → project save/load.

- **Bilingual UI**: switch between 中文 and English with the selector in the top-right header.
  The choice is remembered in the browser; the first visit follows the browser language.
  Element Plus widgets (dialogs, date/number controls, etc.) follow the same locale.
- **Charts**: rendered with Plotly.js. EEMs are drawn as filled contour plots with contour lines
  (no smoothing), with an adjustable contour-level count (0 = automatic).
  A 3D surface view is available on the preprocessing page.
- Every chart can be exported as PNG / JPEG / SVG; the raster resolution is chosen as explicit pixel
  dimensions (e.g. 844 × 784). Legends are shown by default and each chart has a legend on/off switch.
- The FRI plot uses Em on the x-axis and Ex on the y-axis, with the five regions and Ⅰ–Ⅴ labels drawn.
- The history timeline supports restoring any step and deleting a step together with all later steps.
- Inner filter correction, blank subtraction and Raman normalisation can only be applied once;
  both the UI and the API reject repeated execution. Use the history timeline to undo first.
- Missing values remaining after interpolation are reported and treated as 0 during PARAFAC fitting.
- The import page offers a demo zip (`eem/`, `abs/`, `blank/`, `SampleLog.xlsx`) so the full workflow
  can be tried immediately.
- The server listens on `127.0.0.1` only and has no user system; it is intended for single-machine use.

### Navigation

1. **Data import** – upload folders/files or a zip package, or load an existing project.
2. **Preprocessing** – all correction steps with live preview and a history timeline.
3. **PARAFAC analysis** – separate pages for 3.1 modelling, 3.2 outlier detection, 3.3 split-half
   validation and 3.4 results & export.
4. **Indices / FRI / slope** – result tables, charts and Excel downloads.
5. **Project save & export** – save/load `.eemproj` projects and download selected results.

### Building the frontend (only when developing from source)

```bash
cd frontend
npm install
npm run build
```

The build output is written to `src/EEMs_toolkit/web/static/` and bundled into the wheel.
If this directory is missing, `eems-web` prints a hint.

## Development environment

Windows 10, Python 3.11.7. Main dependencies:

|   Package  | Version |
| :--------: | :-----: |
|    numpy   |  2.4.6  |
|   pandas   |  3.1.1  |
| matplotlib |  3.11.2 |
|    scipy   |  1.17.1 |
|  tensorly  |  0.10.0 |
|    xlrd    |  2.0.2  |
|  openpyxl  |  3.1.5  |

Web frontend: Vue 3, TypeScript, Element Plus, Plotly.js, vue-i18n.

## Examples

See `demo1.ipynb` and the `Example/` folder (`Example/demo.xlsx` contains 14 samples).

## GUI tutorial (releases)

[EEMs-toolkit GUI video tutorial (Chinese)](https://zhuanlan.zhihu.com/p/605396414)

## Cite

Machine Learning-Assisted Source Tracing in Domestic-Industrial Wastewater: A Fluorescence
Information-Based Approach
DOI: 10.1016/j.watres.2024.122618

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HHhyJJ/EEMs-toolkit\&type=Date)](https://star-history.com/#HHhyJJ/EEMs-toolkit\&Date)
