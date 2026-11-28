import type { LocaleModule } from './types'

const module: LocaleModule = {
  zh: {
    menu: {
      import: '1. 数据导入',
      preprocess: '2. 预处理',
      parafac: '3. PARAFAC 分析',
      model: '3.1 建模',
      outlier: '3.2 异常值识别',
      validation: '3.3 分半验证',
      result: '3.4 结果与导出',
      indices: '4. 指数 / FRI / 斜率',
      export: '5. 项目保存与导出'
    }
  },
  en: {
    menu: {
      import: '1. Data import',
      preprocess: '2. Preprocessing',
      parafac: '3. PARAFAC analysis',
      model: '3.1 Modelling',
      outlier: '3.2 Outlier detection',
      validation: '3.3 Split-half validation',
      result: '3.4 Results & export',
      indices: '4. Indices / FRI / slope',
      export: '5. Project save & export'
    }
  }
}

export default module
