import Plotly from 'plotly.js-dist-min'

export type ChartFormat = 'png' | 'jpeg' | 'svg'

export function saveChart(
  el: HTMLElement | null,
  name: string,
  format: ChartFormat,
  ratio: number
): boolean {
  if (!el) return false
  const rect = el.getBoundingClientRect()
  if (!rect.width || !rect.height) return false
  const width = Math.max(1, Math.round(rect.width * ratio))
  const height = Math.max(1, Math.round(rect.height * ratio))
  Plotly.downloadImage(el, { format, filename: name, width, height, scale: 1 })
  return true
}
