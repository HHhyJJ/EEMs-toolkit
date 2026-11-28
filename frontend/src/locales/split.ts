export default {
  zh: {
    split: {
      validateComponents: '请输入要验证的组分数，例如 2-4',
      component: '组分 {n}',
      title: '3.3 分半验证（S4C6T3）',
      hint: '需要先完成 PARAFAC 建模；可一次验证多个组分数，TCC ≥ 0.95 视为通过。',
      componentCount: '组分数',
      componentPlaceholder: '例如 2-4 或 2,3,4',
      maxIterations: '最大迭代',
      fixedRandomState: '固定 random_state',
      startSplit: '开始分半',
      stop: '停止',
      randomStateHint: '不勾选“固定 random_state”时每次随机划分样本；勾选后可复现实验。',
      tccValidation: 'TCC 验证（f = {f}）',
      minimumTcc: '最小 TCC',
      passed: '通过',
      failed: '未通过',
      comparisonCount: '{count} 组比较',
      empty: '尚未运行分半验证'
    }
  },
  en: {
    split: {
      validateComponents: 'Enter the component numbers to validate, e.g. 2-4.',
      component: 'Component {n}',
      title: '3.3 Split-Half Validation (S4C6T3)',
      hint: 'Complete PARAFAC modeling first. Multiple component numbers can be validated together; TCC ≥ 0.95 passes.',
      componentCount: 'Components',
      componentPlaceholder: 'e.g. 2-4 or 2,3,4',
      maxIterations: 'Max Iterations',
      fixedRandomState: 'Fix random_state',
      startSplit: 'Start Split-Half Validation',
      stop: 'Stop',
      randomStateHint: 'Samples are split randomly on each run unless "Fix random_state" is selected. Select it to reproduce results.',
      tccValidation: 'TCC Validation (f = {f})',
      minimumTcc: 'Minimum TCC',
      passed: 'Passed',
      failed: 'Failed',
      comparisonCount: '{count} comparisons',
      empty: 'Split-half validation has not been run yet'
    }
  }
}
