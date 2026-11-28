<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { baseLayout, usePlotly } from '../charts/plotly'
import ChartActions from './ChartActions.vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

interface OutlierSample {
  order: number
  file: string
  leverage: number
  sse: number
}

const props = defineProps<{
  samples: OutlierSample[]
  height?: string
}>()

const emit = defineEmits<{ (event: 'change', orders: number[]): void }>()

const selected = ref<number[]>([])
const el = ref<HTMLElement | null>(null)
const legendVisible = ref(true)

watch(() => props.samples, () => {
  if (selected.value.length) {
    selected.value = []
    emit('change', [])
  }
})

function trace(samples: OutlierSample[], label: string, color: string, size: number) {
  return {
    type: 'scatter',
    mode: 'markers',
    name: label,
    x: samples.map((sample) => sample.leverage),
    y: samples.map((sample) => sample.sse),
    customdata: samples.map((sample) => [sample.order, sample.file]),
    marker: { color, size, line: { color: '#ffffff', width: 1 } },
    hovertemplate: '%{customdata[1]}<br>Leverage %{x:.4f}<br>SSE %{y:.4f}<extra></extra>'
  }
}

const plotData = computed(() => [
  trace(props.samples.filter((sample) => !selected.value.includes(sample.order)), t('outlierPlot.samples'), '#409eff', 11),
  trace(props.samples.filter((sample) => selected.value.includes(sample.order)), t('outlierPlot.selected'), '#f56c6c', 13)
])

const layout = computed(() => baseLayout({
  xaxis: {
    title: { text: 'Leverage', font: { size: 11 } },
    tickfont: { size: 10 },
    showgrid: true,
    gridcolor: '#f0f0f0',
    zeroline: false,
    automargin: true
  },
  yaxis: {
    title: { text: 'SSE', font: { size: 11 } },
    tickfont: { size: 10 },
    showgrid: true,
    gridcolor: '#f0f0f0',
    zeroline: false,
    automargin: true
  },
  legend: legendVisible.value
    ? { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.28, font: { size: 11 } }
    : undefined,
  showlegend: legendVisible.value,
  margin: { l: 68, r: 24, t: 22, b: legendVisible.value ? 78 : 52 }
}))

usePlotly(el, plotData, layout, {
  onClick: (event: any) => {
    const order = event?.points?.[0]?.customdata?.[0]
    if (order === undefined) return
    selected.value = selected.value.includes(order)
      ? selected.value.filter((item) => item !== order)
      : [...selected.value, order]
    emit('change', selected.value)
  }
})

function clearSelection() {
  selected.value = []
  emit('change', [])
}
</script>

<template>
  <div>
    <div class="chart-wrap">
      <ChartActions :el="el" name="outlier">
        <el-checkbox v-model="legendVisible" size="small">{{ t('charts.legend') }}</el-checkbox>
      </ChartActions>
      <div ref="el" class="chart-box" :style="{ height: height || '380px' }"></div>
    </div>
    <div class="toolbar" style="margin-top: 8px">
      <span style="font-size: 12px; color: #909399">{{ t('outlierPlot.hint') }}</span>
      <el-button size="small" @click="clearSelection">{{ t('outlierPlot.clearSelection') }}</el-button>
      <el-tag v-if="selected.length" type="danger">{{ t('outlierPlot.selectedCount', { count: selected.length }) }}</el-tag>
    </div>
  </div>
</template>
