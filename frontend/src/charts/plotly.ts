import { onBeforeUnmount, onMounted, shallowRef, watch, type Ref } from 'vue'
import Plotly from 'plotly.js-dist-min'

export const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  displayModeBar: 'hover' as const,
  scrollZoom: false,
  modeBarButtonsToRemove: [
    'lasso2d',
    'select2d',
    'autoScale2d',
    'toggleSpikelines',
    'hoverCompareCartesian',
    'hoverClosestCartesian'
  ]
}

export const SERIES_COLORS = [
  '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2'
]

export const FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif'

export function baseLayout(extra: Record<string, any> = {}): Record<string, any> {
  return {
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { family: FONT_FAMILY, size: 11, color: '#303133' },
    margin: { l: 64, r: 26, t: 30, b: 62 },
    hovermode: 'closest',
    ...extra
  }
}

export function usePlotly(
  el: Ref<HTMLElement | null>,
  data: Ref<any[]>,
  layout: Ref<Record<string, any>>,
  handlers: { onClick?: (event: any) => void } = {}
) {
  const ready = shallowRef(false)
  let observer: ResizeObserver | null = null

  async function draw() {
    if (!el.value) return
    await Plotly.react(el.value, data.value, layout.value, PLOT_CONFIG)
    ready.value = true
  }

  onMounted(async () => {
    if (!el.value) return
    await Plotly.newPlot(el.value, data.value, layout.value, PLOT_CONFIG)
    ready.value = true
    if (handlers.onClick) el.value.on('plotly_click', handlers.onClick)
    observer = new ResizeObserver(() => {
      if (el.value) Plotly.Plots.resize(el.value)
    })
    observer.observe(el.value)
  })

  watch([data, layout], draw, { deep: true })

  onBeforeUnmount(() => {
    observer?.disconnect()
    observer = null
    if (el.value) Plotly.purge(el.value)
    ready.value = false
  })

  return { ready }
}
