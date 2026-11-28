<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { baseLayout, usePlotly } from '../charts/plotly'
import ChartActions from './ChartActions.vue'

const { t } = useI18n()

const props = defineProps<{
  ex: number[]
  em: number[]
  z: (number | null)[][]
  zmin?: number
  zmax?: number
  title?: string
  height?: string
  xName?: string
  yName?: string
  contourCount?: number
  regions?: {
    ex?: number[]
    em?: number[]
    labels?: { x: number; y: number; text: string }[]
    lines?: { x1: number; y1: number; x2: number; y2: number }[]
  }
}>()

const el = ref<HTMLElement | null>(null)
const name = computed(() => props.title || 'eem')
const shortX = computed(() => (props.xName ?? 'Ex').replace(' / nm', ''))
const shortY = computed(() => (props.yName ?? 'Em').replace(' / nm', ''))

function nearestValue(values: number[], target: number): number {
  let best = values[0]
  let bestDiff = Number.POSITIVE_INFINITY
  for (const value of values) {
    const diff = Math.abs(value - target)
    if (diff < bestDiff) {
      bestDiff = diff
      best = value
    }
  }
  return best
}

const matrix = computed(() =>
  props.z.map((row) => row.map((value) => (value === null || value === undefined ? NaN : value)))
)

const zRange = computed(() => {
  const flat = matrix.value.flat().filter((value) => Number.isFinite(value))
  const min = props.zmin ?? (flat.length ? Math.min(...flat) : 0)
  let max = props.zmax ?? (flat.length ? Math.max(...flat) : 1)
  if (max === min) max = min + 1
  return { min, max }
})

const plotData = computed(() => {
  const contourOptions = props.contourCount && props.contourCount > 0 ? { ncontours: props.contourCount } : {}
  return [{
    type: 'contour',
    x: props.ex,
    y: props.em,
    z: matrix.value,
    colorscale: 'Viridis',
    zmin: zRange.value.min,
    zmax: zRange.value.max,
    connectgaps: false,
    hoverongaps: false,
    line: { width: 0.5, color: 'rgba(0,0,0,0.28)' },
    contours: { coloring: 'fill', showlines: true, ...contourOptions },
    hovertemplate: `${shortX.value} %{x} nm<br>${shortY.value} %{y} nm<br>%{z:.4g}<extra></extra>`,
    colorbar: {
      orientation: 'h',
      x: 0.5,
      xanchor: 'center',
      y: -0.3,
      yanchor: 'top',
      len: 0.66,
      thickness: 12,
      title: { text: t('charts.intensity'), side: 'top', font: { size: 10 } },
      tickfont: { size: 9 },
      tickformat: '.3~g',
      outlinewidth: 0,
      ticks: 'outside',
      ticklen: 3
    }
  }]
})

const layout = computed(() => {
  const shapes: any[] = []
  for (const value of props.regions?.em ?? []) {
    const y = nearestValue(props.em, value)
    shapes.push({
      type: 'line', xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: y, y1: y,
      line: { color: 'rgba(255,255,255,0.92)', width: 1, dash: 'dash' }
    })
  }
  for (const value of props.regions?.ex ?? []) {
    const x = nearestValue(props.ex, value)
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: x, x1: x, y0: 0, y1: 1,
      line: { color: 'rgba(255,255,255,0.92)', width: 1, dash: 'dash' }
    })
  }
  for (const segment of props.regions?.lines ?? []) {
    shapes.push({
      type: 'line',
      x0: nearestValue(props.ex, segment.x1),
      x1: nearestValue(props.ex, segment.x2),
      y0: nearestValue(props.em, segment.y1),
      y1: nearestValue(props.em, segment.y2),
      line: { color: 'rgba(255,255,255,0.92)', width: 1, dash: 'dash' }
    })
  }
  const annotations = (props.regions?.labels ?? []).map((label) => ({
    x: nearestValue(props.ex, label.x),
    y: nearestValue(props.em, label.y),
    text: label.text,
    showarrow: false,
    font: { color: '#ffffff', size: 14 },
    bgcolor: 'rgba(0,0,0,0.38)',
    borderpad: 3
  }))
  return baseLayout({
    title: props.title
      ? { text: props.title, x: 0.5, xanchor: 'center', font: { size: 13, color: '#303133' } }
      : undefined,
    xaxis: {
      title: { text: props.xName ?? 'Ex / nm', font: { size: 11 } },
      tickfont: { size: 10 },
      nticks: 10,
      showgrid: false,
      zeroline: false,
      automargin: true
    },
    yaxis: {
      title: { text: props.yName ?? 'Em / nm', font: { size: 11 } },
      tickfont: { size: 10 },
      nticks: 9,
      showgrid: false,
      zeroline: false,
      automargin: true
    },
    margin: { l: 70, r: 26, t: props.title ? 42 : 24, b: 84 },
    shapes,
    annotations
  })
})

usePlotly(el, plotData, layout)
</script>

<template>
  <div class="chart-wrap" :class="{ square: !height }">
    <ChartActions :el="el" :name="name" />
    <div ref="el" class="chart-box" :style="height ? { height } : undefined"></div>
  </div>
</template>
