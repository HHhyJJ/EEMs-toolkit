<script setup lang="ts">
import { computed, ref } from 'vue'
import { baseLayout, SERIES_COLORS, usePlotly } from '../charts/plotly'
import ChartActions from './ChartActions.vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps<{
  x: (string | number)[]
  series: { name: string; data: (number | null)[]; dashed?: boolean; color?: string; type?: string; area?: boolean }[]
  xName?: string
  yName?: string
  title?: string
  height?: string
  legend?: boolean
}>()

const el = ref<HTMLElement | null>(null)
const legendVisible = ref(props.legend !== false)
const name = computed(() => props.title || props.yName || 'chart')

function withAlpha(color: string, alpha: number): string {
  if (!color.startsWith('#') || color.length !== 7) return `rgba(31, 119, 180, ${alpha})`
  const r = parseInt(color.slice(1, 3), 16)
  const g = parseInt(color.slice(3, 5), 16)
  const b = parseInt(color.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

const plotData = computed(() => props.series.map((item, index) => {
  const color = item.color ?? SERIES_COLORS[index % SERIES_COLORS.length]
  const x = props.x.map((value) => String(value))
  if (item.type === 'bar') {
    return {
      type: 'bar',
      name: item.name,
      x,
      y: item.data,
      marker: { color },
      hovertemplate: `${item.name}: %{y:.4g}<extra></extra>`
    }
  }
  return {
    type: 'scatter',
    mode: 'lines',
    name: item.name,
    x,
    y: item.data,
    line: { color, width: 1.7, dash: item.dashed ? 'dash' : 'solid' },
    fill: item.area ? 'tozeroy' : 'none',
    fillcolor: item.area ? withAlpha(color, 0.12) : undefined,
    connectgaps: false,
    hovertemplate: `${item.name}: %{y:.4g}<extra></extra>`
  }
}))

const layout = computed(() => baseLayout({
  title: props.title
    ? { text: props.title, x: 0.5, xanchor: 'center', font: { size: 13, color: '#303133' } }
    : undefined,
  xaxis: {
    type: 'category',
    title: { text: props.xName ?? '', font: { size: 11 } },
    tickfont: { size: 10 },
    showgrid: false,
    zeroline: false,
    automargin: true
  },
  yaxis: {
    title: { text: props.yName ?? '', font: { size: 11 } },
    tickfont: { size: 10 },
    showgrid: true,
    gridcolor: '#f0f0f0',
    zeroline: false,
    automargin: true
  },
  legend: legendVisible.value
    ? { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.3, font: { size: 11 } }
    : undefined,
  showlegend: legendVisible.value,
  hovermode: 'x unified',
  margin: {
    l: 68,
    r: 24,
    t: props.title ? 42 : 24,
    b: legendVisible.value ? 86 : 56
  }
}))

usePlotly(el, plotData, layout)
</script>

<template>
  <div class="chart-wrap" :class="{ square: false }">
    <ChartActions :el="el" :name="name">
      <el-checkbox v-model="legendVisible" size="small">{{ t('charts.legend') }}</el-checkbox>
    </ChartActions>
    <div ref="el" class="chart-box" :style="{ height: height || '320px' }"></div>
  </div>
</template>
