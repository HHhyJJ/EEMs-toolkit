import { createI18n } from 'vue-i18n'
import { messages } from './locales'

export type AppLocale = 'zh-CN' | 'en-US'

const stored = localStorage.getItem('eems-locale') as AppLocale | null
const detected: AppLocale = navigator.language.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US'

export const i18n = createI18n({
  legacy: false,
  locale: stored ?? detected,
  fallbackLocale: 'zh-CN',
  messages,
  missingWarn: false,
  fallbackWarn: false
})

export function setLocale(locale: AppLocale) {
  i18n.global.locale.value = locale
  localStorage.setItem('eems-locale', locale)
  document.documentElement.lang = locale
}
