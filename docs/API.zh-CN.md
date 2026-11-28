# EEMs-toolkit — API 参考手册

> [English](API.md) ｜ 本文件由源码自动生成，覆盖 `src/EEMs_toolkit/` 下全部公开模块、类、函数与常量。

自动生成的参考文档，覆盖每个公开模块、类、函数与常量。

## 目录

- 快速上手
- 核心库（脚本编程 API）
  - [顶层包 EEMs_toolkit](#eems-toolkit)
  - [constant 模块（物理常数与波长区域）](#eems-toolkit-constant)
  - [loader 模块（文件读取）](#eems-toolkit-loader)
  - [absorbance 模块（吸光度与光谱斜率）](#eems-toolkit-absorbance)
  - [eem 模块（EEMsDataset 主类）](#eems-toolkit-eem)
  - [parafac 模块（PARAFAC 引擎）](#eems-toolkit-parafac)
  - [export 模块（结果导出构建器）](#eems-toolkit-export)
  - [project 模块（快照与项目归档）](#eems-toolkit-project)
- Web 子包（FastAPI 接口）
  - [web 模块（可选 Web 界面）](#eems-toolkit-web)
  - [web.app（FastAPI 应用工厂）](#eems-toolkit-web-app)
  - [web.\_\_main__（eems-web 入口）](#eems-toolkit-web-main)
  - [web.session（会话与历史时间线）](#eems-toolkit-web-session)
  - [web.jobs（后台任务管理）](#eems-toolkit-web-jobs)
  - [web.payloads（图表 JSON 数据）](#eems-toolkit-web-payloads)
  - [web.schemas（请求/响应模型）](#eems-toolkit-web-schemas)
  - [routers.data（导入、状态、历史、项目）](#eems-toolkit-web-routers-data)
  - [routers.preprocess（预处理接口）](#eems-toolkit-web-routers-preprocess)
  - [routers.parafac（PARAFAC / 分半 / SSE）](#eems-toolkit-web-routers-parafac)
  - [routers.analysis（指数/FRI/斜率/导出）](#eems-toolkit-web-routers-analysis)

### 快速上手

```python
import EEMs_toolkit as et

# 1. 读取数据
eems = et.read_eems(file_list=['s1.xlsx', 's2.xlsx'], eem_path='eem/')
abs_data = et.read_abs(file_list=['s1.xlsx'], abs_path='abs/')
blank = et.read_blank(file_list=['blank.xlsx'], blank_path='blank/')

# 2. 预处理
eems.inner_effect_correct(abs_data)      # 内滤效应校正
eems.minus_the_blank(blank)               # 扣除空白
eems.raman_areal(blank)                   # 拉曼单位化 AU → RU
eems.cut_ray_scatter((10, 10), (10, 10))  # 去除瑞利散射
eems.cut_ram_scatter((10, 10), (10, 10))  # 去除拉曼散射
eems.miss_value_interpolation()           # 缺失值插值

# 3. PARAFAC 建模（同时拟合 2、3 个组分）
eems.multi_non_parafac_cal(f=[2, 3], starts=10, max_iter=2500)
eems.plot_core_consistency_and_explanation()
eems.plot_sse()                           # 不同组分数的 SSE 对比
eems.plot_outlier_test(f=3)

# 4. 分半验证
eems.split_analysis(f=3, starts=10)

# 5. 指数 / FRI / 光谱斜率
eems.fluorescence_indices_output()
eems.fri_result_output()
abs_data.slope_fit()

# 6. 导出
eems.open_fluor(f=3)
eems.parafac_result_output(f=3)
```

FastAPI 接口详见下方 Web HTTP API 一节。


---

## 核心库（脚本编程 API）

## 顶层包 EEMs_toolkit

<a id="eems-toolkit"></a>

**模块**: `EEMs_toolkit`

### 概述

顶层包，重新导出公开的脚本编程 API：EEMsDataset、Abs、read_eems、read_abs 和 read_blank。


## constant 模块（物理常数与波长区域）

<a id="eems-toolkit-constant"></a>

**模块**: `EEMs_toolkit.constant`

### 概述

整个工具包共用的物理常数与波长区域（峰位、指数、拉曼、分半）。

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `BIX` | tuple | `((380, 430), 310)` | 生物源指数区域：Ex 310 下 Em 380 与 430 的比值。 |
| `CROSS` | tuple | `(((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))` | 4 个分组的三种交叉验证配对组合。 |
| `FLUI` | tuple | `((470, 520), 370)` | 荧光指数区域：Ex 370 下 Em 470–520。 |
| `FREI` | tuple | `((380, (420, 430)), 310)` | 新鲜度指数区域：Ex 310 下 Em 380 / (420–430)。 |
| `HIX` | tuple | `(((435, 480), (300, 345)), 254)` | 腐殖化指数区域：Ex 254 下 Em 435–480 与 300–345 的比值。 |
| `LONG_RANGE` | tuple | `(300, 600)` | 斜率拟合默认使用的长波长范围 (300, 600) nm。 |
| `N_SPLIT` | int | `4` | S4C6T3 分半分析使用的样本分组数 (4)。 |
| `PEAKS` | str | `'BTAMCDN'` | 七个经典 EEM 峰标识字符串：B、T、A、M、C、D、N。 |
| `PEAK_A` | tuple | `((400, 460), 260)` | A 峰区域 (Em 400–460, Ex 260)。 |
| `PEAK_B` | tuple | `(305, 275)` | B 峰位置 (Em 305, Ex 275)。 |
| `PEAK_C` | tuple | `((420, 460), (320, 360))` | C 峰区域 (Em 420–460, Ex 320–360)。 |
| `PEAK_D` | tuple | `(509, 390)` | D 峰位置 (Em 509, Ex 390)。 |
| `PEAK_M` | tuple | `((370, 410), (290, 310))` | M 峰区域 (Em 370–410, Ex 290–310)。 |
| `PEAK_N` | tuple | `(370, 280)` | N 峰位置 (Em 370, Ex 280)。 |
| `PEAK_T` | tuple | `(340, 275)` | T 峰位置 (Em 340, Ex 275)。 |
| `RAMAN_EM_RANGE` | tuple | `(378, 424)` | 用于积分拉曼峰的默认发射范围 (378, 424) nm。 |
| `RAMAN_EX` | int | `350` | 拉曼单位化默认使用的激发波长 (350 nm)。 |
| `S_275_295` | tuple | `(275, 295)` | 短波长光谱斜率 S 的波长范围 (275, 295) nm。 |
| `S_350_400` | tuple | `(350, 400)` | 长波长光谱斜率 S 的波长范围 (350, 400) nm。 |
| `WATER_FREQ` | int | `3382` | 水的拉曼位移 (3382 cm⁻¹)，用于定位拉曼散射。 |

## loader 模块（文件读取）

<a id="eems-toolkit-loader"></a>

**模块**: `EEMs_toolkit.loader`

### 概述

EEM、吸光度与空白样本的文件读取器。

#### 函数

- **`read_abs`**
  - `read_abs(file_list: 'Iterable[str]', abs_path: 'str') -> 'Abs'`
  - 读取吸光度样本。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `file_list` | 吸光度文件名列表。 |
  | `abs_path` | 吸光度文件所在的目录。 |
  - **返回值**: 吸光度数据集。

- **`read_blank`**
  - `read_blank(file_list: 'Iterable[str]', blank_path: 'str') -> 'np.ndarray[tuple[int, int, int], np.dtype[np.float64]]'`
  - 读取空白 EEM 样本，列为不同 Ex、行为不同 Em。表头为 Ex，索引为 Em。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `file_list` | 空白样本文件名列表。 |
  | `blank_path` | 空白 EEM 文件所在的目录。 |
  - **返回值**: 空白 EEM

- **`read_eems`**
  - `read_eems(file_list: 'Iterable[str]', eem_path: 'str') -> 'EEMsDataset'`
  - 读取 EEM 样本，列为不同 Ex、行为不同 Em。表头为 Ex，索引为 Em。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `file_list` | EEM 文件名列表。 |
  | `eem_path` | EEM 文件所在的目录。 |
  - **返回值**: EEMsDataset。

## absorbance 模块（吸光度与光谱斜率）

<a id="eems-toolkit-absorbance"></a>

**模块**: `EEMs_toolkit.absorbance`

### 概述

吸光度数据集容器与光谱斜率拟合。

#### 类

##### `Abs`
吸光度数据集对象。

###### 方法

- **`check_wavelength`**
  - `check_wavelength(self, wave: 'float') -> 'bool'`

- **`slope_fit`**
  - `slope_fit(self, long_range: 'tuple[int, int]' = (300, 600)) -> 'pd.DataFrame'`
  - 在 CDOM 吸收光谱的两个不同区间拟合光谱斜率。会在当前目录生成名为 'slope_fit_result.xlsx' 的 Excel 文件。指数模型：y = a350*exp(S/1000*(350-lambda))+k。参考文献 DOI：10.4319/lo.2008.53.3.0955，10.4319/lo.2001.46.8.2087。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `long_range` | 长波长斜率的波长范围；默认 (300, 600)。 |
  - **返回值**: 结果的 DataFrame。

## eem 模块（EEMsDataset 主类）

<a id="eems-toolkit-eem"></a>

**模块**: `EEMs_toolkit.eem`

### 概述

EEMsDataset 类：预处理、PARAFAC 建模、荧光指数/FRI 以及绘图。

#### 类

##### `EEMsDataset`
EEM 数据集对象。

###### 方法

- **`copy`**
  - `copy(self) -> "'EEMsDataset'"`
  - 返回该数据集的独立副本。

- **`cut_ram_scatter`**
  - `cut_ram_scatter(self, first: 'tuple[float, float]', second: 'tuple[float, float]', freq: 'float' = 3382)`
  - 去除拉曼散射（原地修改），用 NaN 替换。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `first` | 二元组，表示一阶拉曼散射中心线下方与上方的宽度。 |
  | `second` | 二元组，表示二阶拉曼散射中心线下方与上方的宽度。 |
  | `freq` | 可选，拉曼位移常数，水约为 3400-3600 cm⁻¹；默认 3382。 |
- **`cut_ray_scatter`**
  - `cut_ray_scatter(self, first: 'tuple[float, float]', second: 'tuple[float, float]')`
  - 去除瑞利散射（原地修改），用 NaN 替换。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `first` | 二元组，表示一阶瑞利散射中心线下方与上方的宽度。 |
  | `second` | 二元组，表示二阶瑞利散射中心线下方与上方的宽度。 |
- **`dilute`**
  - `dilute(self, dilution_factors: 'Iterable[float]')`
  - 稀释校正（原地修改）。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `dilution_factors` | 每个样本的稀释因子列表。 |
- **`eems_output`**
  - `eems_output(self, output_path: 'Optional[str]' = None)`
  - 导出 EEMs

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `output_path` | 可选，保存文本的目录；不给出时保存在当前目录。 |
- **`fluorescence_indices_output`**
  - `fluorescence_indices_output(self, output_path: 'Optional[str]' = None, output_name: 'str' = 'fluorescence_indices.xlsx')`
  - 计算荧光指数并导出

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `output_path` | 可选，保存文本的目录；不给出时保存在当前目录。 |
- **`fri_result_output`**
  - `fri_result_output(self, output_path: 'Optional[str]' = None, filename: 'str' = 'FRI.xlsx')`
  - 计算 FRI 并导出

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `output_path` | 可选，保存文本的目录；不给出时保存在当前目录。 |
- **`inner_effect_correct`**
  - `inner_effect_correct(self, absorbance: 'Abs')`
  - 内滤效应校正（原地修改）。参考文献 DOI：10.1021/es0155276，10.4319/lom.2013.11.616。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `absorbance` | 一个 Abs 对象。 |
- **`minus_the_blank`**
  - `minus_the_blank(self, blank: 'np.ndarray[tuple[int, int] | tuple[int, int, int], np.dtype[np.float64]]')`
  - 扣除空白样本（原地修改）。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `blank` | 与 self.x 形状相同，或仅为单个样本。 |
- **`miss_value_interpolation`**
  - `miss_value_interpolation(self)`
  - 使用 Delaunay 三角剖分法进行插值（原地修改），旨在去除 EEMs 中的 NaN，使 PARAFAC 程序能正常运行。参考文献 DOI：10.1016/j.marchem.2004.02.006。

- **`multi_non_parafac_cal`**
  - `multi_non_parafac_cal(self, f: 'Optional[int | Iterable[int]]' = None, tol: 'float' = 1e-06, max_iter: 'int' = 2500, starts: 'int' = 10, *, n_processes: 'Optional[int]' = None, progress=None, stop_event=None)`
  - 通过 HALS 进行 'starts' 次非负 CP 分解。采用分层 ALS（交替最小二乘），逐列更新每个因子（一次更新一列，其余列保持固定）。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，可迭代对象，组分数 |
  | `tol` | float，可选，当重构误差的变化小于 tol 时算法停止；默认 1e-6。 |
  | `max_iter` | int，最大迭代次数 |
  | `starts` | int，要拟合的模型数量 |
  | `n_processes` | 可选，工作进程数，默认 min(starts, 8)。 |
  | `progress` | 可选，进度回调，签名为 progress(done, total, f)。 |
  | `stop_event` | 可选，用于取消任务的 threading.Event。 |
- **`open_fluor`**
  - `open_fluor(self, f: 'int', output_path: 'Optional[str]' = None, filename: 'str' = 'model.txt', **kwargs)`
  - 导出可上传到 OpenFluor 进行比对的文本文件。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，组分数 |
  | `output_path` | 可选，保存文本的目录；不给出时保存在当前目录。 |
  | `kwargs` | 可选，包含个人信息 |
- **`parafac_result_output`**
  - `parafac_result_output(self, f: 'int', output_path: 'Optional[str]' = None, filename: 'str' = 'parafac-result.xlsx')`
  - 导出 PARAFAC 结果，包括 FMax、Em 载荷、Ex 载荷和组分。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，组分数 |
  | `output_path` | 可选，保存文本的目录；不给出时保存在当前目录。 |
- **`plot_core_consistency_and_explanation`**
  - `plot_core_consistency_and_explanation(self)`
  - 绘制核一致性与解释率，组分数取自 self.f。

- **`plot_eem_by1`**
  - `plot_eem_by1(self, f: 'Optional[int]' = None, sample_id: 'Optional[np.typing.ArrayLike]' = None, plot_3d: 'bool' = False)`
  - 绘制 EEMs

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | 可选，给出时用因子重构模型绘图而非 self.x；默认 None。 |
  | `sample_id` | 可选，要绘制的样本；不给出时绘制全部；默认 None。 |
  | `plot_3d` | 可选，为 True 时绘制 3D 曲面，否则绘制等高线填充；默认 False。 |
- **`plot_fingers`**
  - `plot_fingers(self, f: 'int')`
  - 在一张图中绘制各组分，并打印每个组分在 Ex 和 Em 上的峰值。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，组分数 |
- **`plot_fmax`**
  - `plot_fmax(self, f: 'int', use_file_list: 'bool' = False)`
  - 绘制 FMax 以查看组分在样本间的变化。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，组分数 |
  | `use_file_list` | 可选，为 True 时 x 轴刻度使用文件名；默认 False。 |
- **`plot_fri`**
  - `plot_fri(self, sample_id: 'Optional[Sequence[int]]' = None)`
  - 绘制 FRI。参考文献 DOI：10.1021/es034354c。

- **`plot_loadings`**
  - `plot_loadings(self, f: 'int')`
  - 绘制组分载荷

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，组分数 |
- **`plot_outlier_test`**
  - `plot_outlier_test(self, f: 'int')`
  - 新建一张图展示异常样本。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，组分数 |
- **`plot_residual_error`**
  - `plot_residual_error(self, f: 'int', sample_id: 'Optional[np.typing.ArrayLike]' = None)`
  - 在一张图中绘制样本、因子重构模型与残差。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，组分数 |
  | `sample_id` | 可选，要绘制的样本；不给出时绘制全部；默认 None。 |
- **`plot_split_result`**
  - `plot_split_result(self, f: 'int')`
  - 绘制分半分析结果

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，组分数 |
- **`plot_sse`**
  - `plot_sse(self, f: 'Optional[int | Iterable[int]]' = None)`
  - 绘制一个或多个组分数的 SSE。第一个子图展示总 SSE 随组分数的变化，其余三个分别展示按样本、按发射波长、按激发波长聚合的 SSE，以便比较不同组分数。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，可迭代对象，组分数；默认为已拟合的模型。 |
- **`raman_areal`**
  - `raman_areal(self, blank: 'np.ndarray[tuple[int, int] | tuple[int, int, int], np.dtype[np.float64]]', ex: 'float' = 350, em_range: 'tuple[float, float]' = (378, 424))`
  - 把任意单位（AU）转换为拉曼单位（RU）。参考文献 DOI：10.1366/000370209788964548。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `blank` | 与 self.x 形状相同，或仅为单个样本。 |
  | `ex` | 可选，拉曼峰的激发波长位置；默认 350。 |
  | `em_range` | 可选，二元组，用于计算面积的范围；不给出时为 (378, 424)。 |
- **`restore`**
  - `restore(self, snapshot: 'dict')`
  - 根据 snapshot 的输出原地恢复可变状态。

- **`smooth_eem`**
  - `smooth_eem(self, sigma: 'float' = 0.5, filter_size: 'Optional[int]' = None)`
  - 使用高斯核对 EEMs 进行平滑（原地修改）。建议在插值之后使用。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `sigma` | 可选，高斯分布的标准差，须为正数；默认 0.5。 |
  | `filter_size` | 可选，高斯滤波器的大小，须为正奇数；默认 2\*ceil(2\*sigma)+1。 |
- **`snapshot`**
  - `snapshot(self) -> 'dict'`
  - 返回可变状态的深拷贝（供历史时间线使用）。

- **`split_analysis`**
  - `split_analysis(self, f: 'Optional[int | Iterable[int]]' = None, tol: 'float' = 1e-06, max_iter: 'int' = 2500, starts: 'int' = 10, random_state: 'Optional[int]' = None, *, n_processes: 'Optional[int]' = None, progress=None, stop_event=None)`
  - 在数据集分半中生成嵌套的 PARAFAC 模型。TCC 大于 0.95 可判定通过验证。参考文献 DOI：10.1027/1614-2241.2.2.57。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，组分数 |
  | `random_state` | {None（默认）, int, np.random.RandomState} |
  | `starts` | int，要拟合的模型数量 |
  | `n_processes` | 可选，工作进程数，默认 min(starts, 8)。 |
  | `progress` | 可选，进度回调，签名为 progress(done, total, f)。 |
  | `stop_event` | 可选，用于取消任务的 threading.Event。 |
- **`split_validation`**
  - `split_validation(self, f: 'Optional[int | Iterable[int]]' = None)`

- **`sse_analysis`**
  - `sse_analysis(self, f: 'Optional[int | Iterable[int]]' = None) -> 'dict[int, dict]'`
  - 计算已拟合模型的平方误差和（SSE）。平方误差张量分别沿三个轴聚合：按样本、按发射波长、按激发波长。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `f` | int，可迭代对象，组分数；默认为已拟合的模型。 |
  - **返回值**: {f: {'total': float, 'sample': ndarray, 'em': ndarray, 'ex': ndarray}}

- **`sub_dataset`**
  - `sub_dataset(self, sample_orders: 'Optional[list[int]]' = None, ex_orders: 'Optional[np.ndarray[tuple[int], np.dtype[np.bool_]]]' = None, em_orders: 'Optional[np.ndarray[tuple[int], np.dtype[np.bool_]]]' = None)`
  - 移除样本和/或波长（原地修改）。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `sample_orders` | 正整数列表，1 表示第一个样本。 |
  | `ex_orders` | 例如 self.ex == 240 |
  | `em_orders` | 例如 self.em > 600 |
## parafac 模块（PARAFAC 引擎）

<a id="eems-toolkit-parafac"></a>

**模块**: `EEMs_toolkit.parafac`

### 概述

底层 PARAFAC 引擎：非负 CP（HALS）、分半拟合以及 Tucker 一致性验证。

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `ProgressCallback` | _CallableGenericAlias | `typing.Callable[[int, int, int], NoneType]` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 类

##### `FitResult`
多组分数模型的拟合结果。

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `factors` | `dict[int, CPTensor]` | `{}` |
| `leverage` | `dict[int, np.ndarray]` | `{}` |
| `sse` | `dict[int, np.ndarray]` | `{}` |
| `core_consistency` | `dict[int, float]` | `{}` |
| `explanation_rate` | `dict[int, float]` | `{}` |
| `log` | `list[str]` | `[]` |

##### `JobCancelled`
当长时间运行的拟合被调用方停止时抛出。
- 基类: `Exception`

##### `SplitResult`
分半分析的结果。

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `split_result` | `dict[int, list[CPTensor]]` | `{}` |
| `splits_index` | `list[np.ndarray]` | `[]` |
| `log` | `list[str]` | `[]` |
#### 函数

- **`component_numbers`**
  - `component_numbers(f: 'Optional[int | Iterable[int]]', default: 'Iterable[int]' = range(2, 6)) -> 'tuple[int, ...]'`
  - 把请求的组分数规范化为一个元组。

- **`fit_models`**
  - `fit_models(x: 'np.ndarray', fs: 'Iterable[int]', *, starts: 'int' = 10, tol: 'float' = 1e-06, max_iter: 'int' = 2500, n_processes: 'Optional[int]' = None, progress: 'Optional[ProgressCallback]' = None, stop_event=None) -> 'FitResult'`
  - 对每个请求的组分数进行多次随机起始拟合，并保留最优模型。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `x` | 形状为 (nSample, nem, nex) 的 EEM 数据。 |
  | `fs` | 要拟合的组分数。 |
  | `starts` | 每个组分数的随机起始次数。 |
  | `tol` | HALS 算法的收敛容差。 |
  | `max_iter` | 单次拟合的最大迭代次数。 |
  | `n_processes` | 工作进程数，默认 min(starts, 8)。 |
  | `progress` | 可选，进度回调，签名为 progress(done, total, f)。 |
  | `stop_event` | 可选，用于取消任务的 threading.Event。 |
- **`process_count`**
  - `process_count(starts: 'int', n_processes: 'Optional[int]' = None) -> 'int'`
  - 工作进程数；当 starts 大于 8 时上限为 8。

- **`split_fit`**
  - `split_fit(x: 'np.ndarray', factors: 'dict[int, CPTensor]', fs: 'Iterable[int]', *, starts: 'int' = 10, tol: 'float' = 1e-06, max_iter: 'int' = 2500, random_state: 'Optional[int]' = None, n_processes: 'Optional[int]' = None, progress: 'Optional[ProgressCallback]' = None, stop_event=None) -> 'SplitResult'`
  - 在数据集分半中生成嵌套的 PARAFAC 模型（S4C6T3）。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `x` | 形状为 (nSample, nem, nex) 的 EEM 数据。 |
  | `factors` | 整体模型，用于对齐组分量级的顺序。 |
  | `fs` | 要拟合的组分数。 |
  | `random_state` | 可选，样本打乱的随机种子。 |
- **`split_validation`**
  - `split_validation(factors: 'dict[int, CPTensor]', split_result: 'dict[int, list[CPTensor]]', fs: 'Iterable[int]') -> 'dict[int, dict]'`
  - 使用 Tucker 一致性系数（TCC）比较分半模型与整体模型。

  - **返回值**: 组分数到 {min_tcc, validated, em_tcc, ex_tcc} 的映射。

## export 模块（结果导出构建器）

<a id="eems-toolkit-export"></a>

**模块**: `EEMs_toolkit.export`

### 概述

把数据集结果转换为字节的构建器，供文件写入与 Web API 共用。

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `DEFAULT_EXPORTS` | tuple | `('openfluor', 'parafac', 'eems', 'indices', 'fri', 'slope')` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 函数

- **`all_zip_bytes`**
  - `all_zip_bytes(dataset, abs_obj=None, f: 'Optional[int]' = None, include: 'Optional[Iterable[str]]' = None, **kwargs) -> 'bytes'`
  - 把所有请求的导出项打包为一个 zip 压缩包。不可用的条目会写入 skipped.txt，而不会让整个压缩包失败。

- **`eem_xlsx_bytes`**
  - `eem_xlsx_bytes(dataset, index: 'int') -> 'bytes'`
  - 构建一个 EEM 工作簿，第一行为 Ex，第一列为 Em。

- **`eems_zip_bytes`**
  - `eems_zip_bytes(dataset) -> 'bytes'`
  - 把所有 EEM 工作簿打包为一个 zip 压缩包。

- **`fri_xlsx_bytes`**
  - `fri_xlsx_bytes(dataset) -> 'bytes'`
  - 构建 FRI 工作簿；当波长范围未覆盖时抛出 ValueError。

- **`indices_xlsx_bytes`**
  - `indices_xlsx_bytes(dataset) -> 'bytes'`
  - 构建荧光指数工作簿。

- **`open_fluor_text`**
  - `open_fluor_text(dataset, f: 'int', **kwargs) -> 'bytes'`
  - 构建 OpenFluor 文本文件内容。

- **`parafac_xlsx_bytes`**
  - `parafac_xlsx_bytes(dataset, f: 'int') -> 'bytes'`
  - 构建 PARAFAC 结果工作簿（FMax、载荷与组分图）。

- **`slope_xlsx_bytes`**
  - `slope_xlsx_bytes(abs_obj) -> 'bytes'`
  - 构建吸光度光谱斜率工作簿。

## project 模块（快照与项目归档）

<a id="eems-toolkit-project"></a>

**模块**: `EEMs_toolkit.project`

### 概述

快照辅助函数以及 .eemproj 项目归档格式。

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `FORMAT_VERSION` | int | `1` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 类

##### `LoadedProject`
从 .eemproj 归档中恢复的全部内容。

###### 字段

| 名称 | 类型 | 值 |
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
#### 函数

- **`apply_snapshot`**
  - `apply_snapshot(dataset, snapshot: 'dict') -> 'None'`
  - 根据 take_snapshot 的输出原地恢复数据集。

- **`dump_project`**
  - `dump_project(fileobj, dataset, *, history: 'Optional[list[dict]]' = None, log: 'Optional[list[dict]]' = None, selected_f: 'Optional[int]' = None, sources: 'Optional[dict]' = None, dilution: 'Optional[np.ndarray]' = None, abs_obj=None, blank: 'Optional[np.ndarray]' = None, pipeline: 'Optional[list[dict]]' = None, history_index: 'int' = 0, include_history: 'bool' = True) -> 'None'`
  - 写入 .eemproj 归档。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `fileobj` | 文件路径或二进制文件对象。 |
  | `history` | 步骤元数据；每条记录可包含指向快照 npz 的 path。 |
  | `log` | 界面中显示的流程日志条目。 |
- **`load_project`**
  - `load_project(fileobj, *, history_dir: 'Optional[str]' = None) -> 'LoadedProject'`
  - 读取 .eemproj 归档。

  **参数**

  | 参数 | 说明 |
  |---|---|
  | `fileobj` | 文件路径或二进制文件对象。 |
  | `history_dir` | 可选，接收解压出的历史快照的目录。 |
- **`load_snapshot`**
  - `load_snapshot(fileobj) -> 'dict'`
  - 读取由 save_snapshot 写入的快照。

- **`rebuild_cp_tensor`**
  - `rebuild_cp_tensor(weights: 'np.ndarray', sample: 'np.ndarray', em: 'np.ndarray', ex: 'np.ndarray') -> 'CPTensor'`
  - 按本包使用的因子顺序（样本、发射、激发）构造 CPTensor。

- **`save_snapshot`**
  - `save_snapshot(fileobj, snapshot: 'dict') -> 'None'`
  - 把快照保存为 npz 归档（路径或二进制文件对象）。

- **`take_snapshot`**
  - `take_snapshot(dataset) -> 'dict'`
  - 深拷贝数据集的可变状态。

---

## Web 子包（FastAPI 接口）

## web 模块（可选 Web 界面）

<a id="eems-toolkit-web"></a>

**模块**: `EEMs_toolkit.web`

### 概述

EEMs-toolkit 的可选本地 Web 界面。


## web.app（FastAPI 应用工厂）

<a id="eems-toolkit-web-app"></a>

**模块**: `EEMs_toolkit.web.app`

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `COOKIE_NAME` | str | `'eems_sid'` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
| `STATIC_DIR` | WindowsPath | `` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
| `app` | FastAPI | `` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 函数

- **`create_app`**
  - `create_app() -> 'FastAPI'`

- **`lifespan`**
  - `lifespan(_: 'FastAPI') -> 'AsyncGenerator[None]'`

## web.\_\_main\_\_（eems-web 入口）

<a id="eems-toolkit-web-main"></a>

**模块**: `EEMs_toolkit.web.__main__`

### 概述

eems-web 入口：启动 uvicorn 并打开浏览器。

#### 函数

- **`main`**
  - `main(argv: 'list[str] | None' = None) -> 'None'`

## web.session（会话与历史时间线）

<a id="eems-toolkit-web-session"></a>

**模块**: `EEMs_toolkit.web.session`

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `MAX_HISTORY` | int | `50` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
| `MAX_SESSIONS` | int | `20` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
| `SESSION_TTL` | int | `14400` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
| `STORE` | SessionStore | `` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 类

##### `HistoryLimitReached`
当时间线已达到 MAX_HISTORY 步时抛出。
- 基类: `Exception`

##### `Session`
一个浏览器会话的状态：数据集、时间线以及导入来源。

###### 方法

- **`close`**
  - `close(self) -> 'None'`

- **`invalidate_models`**
  - `invalidate_models(self) -> 'None'`

- **`jump`**
  - `jump(self, index: 'int') -> 'None'`

- **`previous_snapshot`**
  - `previous_snapshot(self) -> 'Optional[dict]'`
  - 当前位置前一步的快照，用于前后对比预览。

- **`record`**
  - `record(self, action: 'str', params: 'Optional[dict]' = None) -> 'None'`
  - 在一次成功的步骤之后，把当前数据集状态追加到历史时间线。

- **`replace_from_project`**
  - `replace_from_project(self, loaded) -> 'None'`
  - 接管一个已载入的项目，不再次复制其数据。

- **`set_dataset`**
  - `set_dataset(self, dataset, *, sources=None, abs_obj=None, blank=None, dilution=None) -> 'None'`

- **`touch`**
  - `touch(self) -> 'None'`

- **`trim`**
  - `trim(self, index: 'int') -> 'None'`
  - 删除 index 处的步骤及其之后的所有步骤。

##### `SessionStore`
有容量上限、带过期时间的会话注册表。

###### 方法

- **`cleanup`**
  - `cleanup(self) -> 'None'`

- **`close_all`**
  - `close_all(self) -> 'None'`

- **`get`**
  - `get(self, sid: 'Optional[str]') -> 'Optional[Session]'`

- **`get_or_create`**
  - `get_or_create(self, sid: 'Optional[str]' = None) -> 'Session'`
#### 函数

- **`get_session`**
  - `get_session(request: 'Request') -> 'Session'`
  - 返回中间件绑定的会话的 FastAPI 依赖。

## web.jobs（后台任务管理）

<a id="eems-toolkit-web-jobs"></a>

**模块**: `EEMs_toolkit.web.jobs`

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `MANAGER` | JobManager | `` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 类

##### `Job`
后台任务的数据结构（dataclass），记录任务 id、类型、会话、进度、日志、起止时间等。

###### 字段

| 名称 | 类型 | 值 |
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

###### 方法

- **`add_log`**
  - `add_log(self, line: 'str') -> 'None'`

- **`progress`**
  - `progress(self, done: 'int', total: 'int', f: 'int') -> 'None'`

- **`snapshot`**
  - `snapshot(self) -> 'dict'`

##### `JobManager`
每个会话同时只运行一个长任务。

###### 方法

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

## web.payloads（图表 JSON 数据）

<a id="eems-toolkit-web-payloads"></a>

**模块**: `EEMs_toolkit.web.payloads`

### 概述

把数据集数组转换为供 Plotly 图表使用的纯 JSON 数据。

#### 函数

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

## web.schemas（请求/响应模型）

<a id="eems-toolkit-web-schemas"></a>

**模块**: `EEMs_toolkit.web.schemas`

#### 类

##### `CutScatterRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `ray` | `tuple[float, float]` | `` |
| `second_ray` | `tuple[float, float]` | `` |
| `ram` | `tuple[float, float]` | `` |
| `second_ram` | `tuple[float, float]` | `` |

##### `DiluteRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `factors` | `typing.Optional[list[float]]` | `` |

##### `ExcludeSamplesRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `orders` | `list[int]` | `` |

##### `ExportAllRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `include` | `typing.Optional[list[str]]` | `` |
| `f` | `typing.Optional[int]` | `` |

##### `HistoryJumpRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `index` | `int` | `` |

##### `ParafacJobRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `f` | `list[int]` | `` |
| `starts` | `int` | `10` |
| `tol` | `float` | `1e-06` |
| `max_iter` | `int` | `2500` |
| `n_processes` | `typing.Optional[int]` | `` |

##### `ProjectSaveRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `include_history` | `bool` | `True` |
| `f` | `typing.Optional[int]` | `` |

##### `RamanRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `ex` | `float` | `350` |
| `em_range` | `tuple[float, float]` | `(378, 424)` |

##### `SmoothRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `sigma` | `float` | `0.5` |
| `filter_size` | `typing.Optional[int]` | `` |

##### `SplitJobRequest`
- 基类: `ParafacJobRequest`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `f` | `list[int]` | `` |
| `starts` | `int` | `10` |
| `tol` | `float` | `1e-06` |
| `max_iter` | `int` | `2500` |
| `n_processes` | `typing.Optional[int]` | `` |
| `random_state` | `typing.Optional[int]` | `` |

##### `SubDatasetRequest`
- 基类: `BaseModel`

###### 字段

| 名称 | 类型 | 值 |
|---|---|---|
| `sample_orders` | `typing.Optional[list[int]]` | `` |
| `ex_lt` | `typing.Optional[float]` | `` |
| `ex_gt` | `typing.Optional[float]` | `` |
| `em_lt` | `typing.Optional[float]` | `` |
| `em_gt` | `typing.Optional[float]` | `` |

## routers.data（导入、状态、历史、项目）

<a id="eems-toolkit-web-routers-data"></a>

**模块**: `EEMs_toolkit.web.routers.data`

### 概述

数据导入、会话状态、历史时间线以及项目归档。

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `DEMO_PACKAGE` | WindowsPath | `EEMs_toolkit/web/demo/eems-demo.zip')` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
| `router` | APIRouter | `` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 函数

- **`demo_package`**
  - `demo_package()`
  - 下载文档中使用的示例压缩包（EEM / 吸光 / 空白 / SampleLog）。

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
#### HTTP 路由

| 方法 | 路径 | 处理函数 | 说明 |
|---|---|---|---|
| GET | `/api/state` | `get_state` |  |
| GET | `/api/samples` | `get_samples` |  |
| POST | `/api/import` | `import_files` |  |
| GET | `/api/import/demo-package` | `demo_package` | 下载文档中使用的示例压缩包（EEM / 吸光 / 空白 / SampleLog）。 |
| GET | `/api/history` | `get_history` |  |
| POST | `/api/history/jump` | `jump_history` |  |
| POST | `/api/history/trim` | `trim_history` |  |
| POST | `/api/project/save` | `save_project` |  |
| POST | `/api/project/load` | `load_project` |  |

## routers.preprocess（预处理接口）

<a id="eems-toolkit-web-routers-preprocess"></a>

**模块**: `EEMs_toolkit.web.routers.preprocess`

### 概述

预处理接口：每次调用都会在会话时间线上追加一个步骤。

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `router` | APIRouter | `` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 函数

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
#### HTTP 路由

| 方法 | 路径 | 处理函数 | 说明 |
|---|---|---|---|
| POST | `/api/preprocess/inner_effect_correct` | `inner_effect_correct` |  |
| POST | `/api/preprocess/minus_blank` | `minus_blank` |  |
| POST | `/api/preprocess/raman_areal` | `raman_areal` |  |
| POST | `/api/preprocess/sub_dataset` | `sub_dataset` |  |
| POST | `/api/preprocess/dilute` | `dilute` |  |
| POST | `/api/preprocess/smooth` | `smooth` |  |
| POST | `/api/preprocess/cut_scatter` | `cut_scatter` |  |
| POST | `/api/preprocess/interpolate` | `interpolate` |  |

## routers.parafac（PARAFAC / 分半 / SSE）

<a id="eems-toolkit-web-routers-parafac"></a>

**模块**: `EEMs_toolkit.web.routers.parafac`

### 概述

PARAFAC / 分半任务（启动、停止、SSE）以及图表数据。

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `router` | APIRouter | `` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 函数

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
#### HTTP 路由

| 方法 | 路径 | 处理函数 | 说明 |
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

## routers.analysis（指数/FRI/斜率/导出）

<a id="eems-toolkit-web-routers-analysis"></a>

**模块**: `EEMs_toolkit.web.routers.analysis`

### 概述

分析面板（荧光指数、FRI、光谱斜率）、异常样本剔除以及导出中心。

#### 常量

| 名称 | 类型 | 值 | 说明 |
|---|---|---|---|
| `XLSX_MEDIA` | str | `'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
| `router` | APIRouter | `` | Web 子包的内部常量，不属于公开的脚本编程 API。 |
#### 函数

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
#### HTTP 路由

| 方法 | 路径 | 处理函数 | 说明 |
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
