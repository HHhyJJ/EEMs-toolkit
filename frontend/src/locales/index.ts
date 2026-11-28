import common from './common'
import menu from './menu'
import components from './components'
import importData from './importData'
import preprocess from './preprocess'
import parafac from './parafac'
import split from './split'
import indices from './indices'
import exportPage from './exportPage'
import type { LocaleModule } from './types'

const modules: LocaleModule[] = [
  common,
  menu,
  components,
  importData,
  preprocess,
  parafac,
  split,
  indices,
  exportPage
]

function collect(lang: 'zh' | 'en') {
  return modules.reduce<Record<string, unknown>>((acc, module) => {
    Object.assign(acc, module[lang])
    return acc
  }, {})
}

export const messages = {
  'zh-CN': collect('zh'),
  'en-US': collect('en')
}
