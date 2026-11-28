import axios from 'axios'
import { i18n } from './i18n'

const api = axios.create({ baseURL: '' })

export function errorMessage(error: any): string {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((i) => i.msg ?? JSON.stringify(i)).join('; ')
  return error?.message ?? i18n.global.t('common.requestFailed')
}

export async function downloadPost(url: string, payload: any, filename: string): Promise<void> {
  const response = await api.post(url, payload, { responseType: 'blob' })
  triggerDownload(response.data, filename)
}

export function downloadUrl(url: string, filename: string): void {
  triggerDownloadFromUrl(url, filename)
}

function triggerDownload(blob: Blob, filename: string): void {
  const objectUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = objectUrl
  link.download = filename
  link.click()
  URL.revokeObjectURL(objectUrl)
}

function triggerDownloadFromUrl(url: string, filename: string): void {
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.target = '_blank'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

export default api
