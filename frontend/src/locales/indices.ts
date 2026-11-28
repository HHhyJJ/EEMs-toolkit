export default {
  zh: {
    indices: {
      pageTitle: '4. 指数 / FRI / 吸收斜率',
      pageHint: '与 PARAFAC 建模相互独立，预处理完成后即可查看。FRI 需要保留 Ex < 250 nm 与 Em 覆盖 330–380 nm 的原始波长范围。',
      fluorescenceIndices: '荧光指数',
      chartMetric: '图表指标',
      downloadExcel: '下载 Excel',
      friUnavailable: '当前数据不满足 FRI 的波长范围（需要 Ex < 250 nm，Em 覆盖 330–380 nm）。',
      sample: '样本',
      contours: '等高线',
      auto: '0 = 自动',
      chartTitle: '样本 {order}: {file}',
      noFriResults: '暂无 FRI 结果',
      spectralSlope: '吸收斜率',
      slopeUnavailable: '未导入吸光数据，无法计算吸收斜率。',
      loadIndicesFailed: '荧光指数加载失败：{message}',
      loadFriFailed: 'FRI 加载失败：{message}',
      loadSlopeFailed: '吸收斜率加载失败：{message}',
      region1: 'Ⅰ 类酪氨酸',
      region2: 'Ⅱ 类色氨酸',
      region3: 'Ⅲ 类富里酸',
      region4: 'Ⅳ 微生物蛋白/溶解性',
      region5: 'Ⅴ 类腐殖酸'
    }
  },
  en: {
    indices: {
      pageTitle: '4. Indices / FRI / Spectral Slope',
      pageHint: 'Independent of PARAFAC modeling and available after preprocessing. FRI requires the original wavelength range to retain Ex < 250 nm and Em covering 330–380 nm.',
      fluorescenceIndices: 'Fluorescence Indices',
      chartMetric: 'Chart Index',
      downloadExcel: 'Download Excel',
      friUnavailable: 'The current data do not meet the wavelength range required for FRI (Ex < 250 nm and Em covering 330–380 nm are required).',
      sample: 'Sample',
      contours: 'Contours',
      auto: '0 = Auto',
      chartTitle: 'Sample {order}: {file}',
      noFriResults: 'No FRI results',
      spectralSlope: 'Spectral Slope',
      slopeUnavailable: 'No absorbance data imported; spectral slope cannot be calculated.',
      loadIndicesFailed: 'Failed to load fluorescence indices: {message}',
      loadFriFailed: 'Failed to load FRI: {message}',
      loadSlopeFailed: 'Failed to load spectral slope: {message}',
      region1: 'Ⅰ Tyrosine-like',
      region2: 'Ⅱ Tryptophan-like',
      region3: 'Ⅲ Fulvic acid-like',
      region4: 'Ⅳ Microbial protein/SMP-like',
      region5: 'Ⅴ Humic acid-like'
    }
  }
}
