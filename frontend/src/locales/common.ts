import type { LocaleModule } from './types'

const module: LocaleModule = {
  zh: {
    app: {
      subtitle: '三维荧光 PARAFAC 工具 · Web 版',
      noData: '未导入数据',
      samples: '样本',
      history: '历史',
      language: '语言'
    },
    common: {
      confirm: '确定',
      cancel: '取消',
      save: '保存',
      delete: '删除',
      download: '下载',
      close: '关闭',
      success: '操作成功',
      failed: '操作失败',
      requestFailed: '请求失败',
      tip: '提示',
      warning: '警告',
      auto: '自动',
      optional: '可选',
      sample: '样本',
      loading: '加载中',
      none: '暂无',
      reset: '重置',
      export: '导出',
      enabled: '已启用',
      disabled: '已禁用',
      chooseFile: '选择文件',
      chooseFiles: '选择文件',
      noFileChosen: '未选择任何文件',
      filesChosen: '已选择 {n} 个文件',
      connectionLost: '与后台任务的连接已断开，请刷新页面查看结果。'
    }
  },
  en: {
    app: {
      subtitle: 'EEM PARAFAC toolkit · Web',
      noData: 'No data imported',
      samples: 'samples',
      history: 'History',
      language: 'Language'
    },
    common: {
      confirm: 'OK',
      cancel: 'Cancel',
      save: 'Save',
      delete: 'Delete',
      download: 'Download',
      close: 'Close',
      success: 'Success',
      failed: 'Operation failed',
      requestFailed: 'Request failed',
      tip: 'Tip',
      warning: 'Warning',
      auto: 'Auto',
      optional: 'Optional',
      sample: 'Sample',
      loading: 'Loading',
      none: 'None',
      reset: 'Reset',
      export: 'Export',
      enabled: 'Enabled',
      disabled: 'Disabled',
      chooseFile: 'Choose file',
      chooseFiles: 'Choose files',
      noFileChosen: 'No file chosen',
      filesChosen: '{n} file(s) selected',
      connectionLost: 'Lost the connection to the background job. Refresh the page to check the result.'
    }
  }
}

export default module
