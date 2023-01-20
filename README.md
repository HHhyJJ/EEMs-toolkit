# **EEMs-toolkit**

EEMs-toolkit是一个可以在Python上对三维荧光（EEM）进行平行因子分析（PARAFAC）的工具包，功能大致类似于MATLAB的drEEM toolbox。





## 开发环境

Windows 10，Python 3.9.6，采用的额外的包及版本见下表：

|     包     |  版本  |
| :--------: | :----: |
|   numpy    | 1.24.1 |
|   pandas   | 1.5.2  |
| matplotlib | 3.6.2  |
|   scipy    | 1.9.3  |
|  tensorly  | 0.7.0  |
|   tlviz    | 0.1.6  |
|   joblib   | 1.2.0  |

------
## 安装EEMs-toolkit
```python
pip install EEMs-toolkit
```


## 功能简介

1. 数据导入

   - 样本日志导入
   - 样本导入
   - 空白导入
   - 紫外光谱导入

2. 预处理

   - 扣除空白

     原始数据：

     ![image-20230119210013034](/Example/pictures/image-20230119210013034.png)

     

     去除空白：

     ![image-20230119210117607](/Example/pictures/image-20230119210117607.png)

   - 去除不需要的波长部分

     ![image-20230119210214514](/Example/pictures/image-20230119210214514.png)

   - 拉曼单位化

     ![image-20230119210258801](/Example/pictures/image-20230119210258801.png)

   - 稀释因子校正

   - 内滤效应校正

     ![image-20230119210344367](/Example/pictures/image-20230119210344367.png)

   - 去除瑞丽（Rayleigh）散射和拉曼（Raman）散射

     ![image-20230119210434896](/Example/pictures/image-20230119210434896.png)

   - 缺失值插值

     ![image-20230119210529378](/Example/pictures/image-20230119210529378.png)

   - 荧光矩阵平滑（高斯核）

3. PARAFAC分析

   - 采用非负约束计算不同组分数的模型

   - 绘制异常样本测试图

     ![image-20230119210653963](/Example/pictures/image-20230119210653963.png)

   - 去除异常样本

   - 重新计算模型，运行分半分析（S4C4T2）

   - 确认组分数（核一致性图，因子相似度图）

     ![image-20230119210738168](/Example/pictures/image-20230119210738168.png)

     ![image-20230119210756988](/Example/pictures/image-20230119210756988.png)

4. 结果展示

   - 核一致性和方差解释率

   - 因子相似度

     ![image-20230119210858476](/Example/pictures/image-20230119210858476.png)

   - 分半分析结果

     ![image-20230119211156840](/Example/pictures/image-20230119211156840.png)

   - 组分图绘制

     ![image-20230119211432900](/Example/pictures/image-20230119211432900.png)

   - 载荷图绘制

     ![image-20230119211450807](/Example/pictures/image-20230119211450807.png)

   - 模型结果导出，包括FMax，Em loadings，Ex loadings，组分图数据

5. 其他功能

   - 预处理后样本数据导出

   - OpenFluor文件导出，可上传对比数据库

   - 区域积分（FRI）计算，结果导出
   - 荧光指数（Fluorescence index、Freshness index、Humification index，Biological index）计算，结果导出
   - 光谱斜率SR计算，结果导出

------



## 测试示例

使用drEEM toolbox中demofiles.zip里的**EEMs、Abs1cm、BlankEEMs**进行演示。**不同代码块代表不同的Python文件**。

1. 数据导入&预处理

   ```python
   # -*- coding = utf-8 -*-
   from EEMs_toolkit import EEMs_Dataset, read_sample_log, read_eems, read_abs, read_blank
   import pickle
   
   """
   读取的目录到时候根据自己文件保存的地方进行修改
   此例文件太多，读取文件耗时，为了不重复读取，将数据读取再预处理后保存到本地
   预处理中内滤效应校正和拉曼单位化可以不用做，取决于自己的需要
   去除散射再插值是为了能够进行PARAFAC，这里不支持包含nan的计算，而不去除散射则会影响计算结果
   建议分别在去除散射和插值之后浏览所有EEM，观察设置去除的散射宽度是否合适
   """
   sample_log = read_sample_log(r'd:\Documents\PycharmProjects\parafac\Example\SampleLog.xlsx')
   Abs, Abs_wave = read_abs(sample_log, r'd:\Documents\PycharmProjects\parafac\Example\abs')
   blank = read_blank(sample_log, r'd:\Documents\PycharmProjects\parafac\Example\blank')
   c, fl = read_eems(d, r'd:\Documents\PycharmProjects\parafac\Example\eem')
   
   x, ex, em = c
   Data = EEMs_Dataset(x, ex, em, file_list=fl)
   Data.plot_eem_by1()
   Data.minus_the_blank(blank)
   # Data.plot_eem_by1()
   Data.raman_areal(blank)
   Data.sub_dataset([], Data.ex < 250, Data.em > 600)
   # Data.plot_eem_by1()
   Data.inner_effect_correct(Abs_wave, Abs)
   # Data.plot_eem_by1()
   Data.cut_ray_scatter([15, 15], [20, 15])
   Data.cut_ram_scatter([15, 15], [10, 10])
   # Data.plot_eem_by1()
   Data.miss_value_interpolation()
   # Data.plot_eem_by1()
   with open('eem_data.pickle', 'wb') as f:
       pickle.dump(Data, f)
   
   ```

2. 开始分析

   ```python
   import pickle
   
   """
   1. 导入刚才预处理好并保存的数据，加载变量
   """
   with open('eem_data.pickle', 'rb') as f:
       data = pickle.load(f)
   """
   2. （浏览所有样本，记录明显存在异常，即与其他样本显著不同或存在测试问题的样本编号）
   """
   # data.plot_eem_by1()
   """
   3. 进行不同组分数的PARAFAC计算
   """
   data.multi_non_parafac_cal([2, 6])
   """
   4. 查看异常样本测试图，尝试不同的组分数，观察其异常样本
   （即位于图右方和上方的样本，冒号前为样本编号），记录其样本编号
   """
   for i in range(2, 6 + 1):
       data.plot_outlier_test(i)
   
   ```

   ```python
   import pickle
   
   """
   5. 去除异常样本后，重新进行不同组分数的PARAFAC计算，此处为简化过程，只去除了部分
   异常样本，主要为看起来显著不同的样本。实际进行操作时建议去除异常样本后重新建立模
   型查看异常样本测试图，反复记录异常样本编号再去除，直到其leverage基本不大于0.3，
   这样后续分析成功率更高，无论去除了多少样本，冒号前面的样本编号不会发生变化，每次
   记录后添加到最开始去除异常样本的位置即可
   """
   with open('eem_data.pickle', 'rb') as f:
       data = pickle.load(f)
   data.sub_dataset([90, 110, 111, 112, 113, 114, 186, 187, 188, 189])
   data.multi_non_parafac_cal([2, 6])
   # for i in range(2, 6 + 1):
   #     data.plot_outlier_test(i)
   """
   6. 后续步骤只有一个目的，确定组分数。首先进行分半分析，这会在窗口中打印出每个组
   分的因子相似度及是否经过相似度检验。也可以画图查看不同组分数的因子相似度，在0.95
   之上可认为是一致的。再综合考虑核一致性（不总是可靠）、不同组分数的模型残差，选择
   相对较好的组分数（因子相似度高、核一致性高、残差为随机分布，不存在明显的峰），
   事实上在这一步一般就会选择满足相似度的最大组分数，而在drEEM中，满足此条件的模型
   可认为是经过验证的，可以结束此过程得到结果了
   """
   data.split_analysis([2, 6])
   data.plot_factor_similarity()
   data.plot_core_consistency_and_explanation()
   for i in range(2, 6 + 1):
       data.plot_residual_error(i)
    """
   因为要分析完前面的结果才能确定组分数，重复跑模型还是很花时间的，所以将跑完的模型
   先保存
   """
   with open('result.pickle', 'wb') as f:
       pickle.dump(data, f)
   
   ```
   
   ```python
   """
   导入已经跑完的模型，下面开始生成结果
   """
   with open('result.pickle', 'rb') as f:
       data = pickle.load(f)
   """
   7. 确定组分数，后面的步骤只剩出图和数据导出（如果需要）：分半分析图、组分图、
   载荷图、（FMax图）
   """
   fac = 6
   #  出图
   data.plot_split_result(fac)
   data.plot_fingers(fac)
   data.plot_loadings(fac)
   # data.plot_fmax(fac)
   #  导出数据
   data.parafac_result_output(fac)
   # data.eems_output()
   # data.open_fluor(fac)
   
   ```

   





