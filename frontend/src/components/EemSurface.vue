<script setup lang="ts">
import { computed, ref } from 'vue'
import { baseLayout, usePlotly } from '../charts/plotly'
import ChartActions from './ChartActions.vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps<{
  ex: number[]
  em: number[]
  z: (number | null)[][]
  height?: string
}>()

const el = ref<HTMLElement | null>(null)

const matrix = computed(() =>
  props.z.map((row) => row.map((value) => (value === null || value === undefined ? NaN : value)))
)

const plotData = computed(() => [{
  type: 'surface',
  x: props.ex,
  y: props.em,
  z: matrix.value,
  colorscale: 'Viridis',
  colorbar: {
    title: { text: 'Intensity', side: 'top', font: { size: 10 } },
    thickness: 12,
    tickfont: { size: 9 },
    tickformat: '.3~g',
    outlinewidth: 0
  },
  contours: {
    z: { show: false }
  },
  hovertemplate: 'Ex %{x} nm<br>Em %{y} nm<br>%{z:.4g}<extra></extra>'
}])

const layout = computed(() => baseLayout({
  margin: { l: 0, r: 0, t: 0, b: 0 },
  scene: {
    xaxis: { title: { text: 'Ex / nm', font: { size: 11 } }, tickfont: { size: 9 } },
    yaxis: { title: { text: 'Em / nm', font: { size: 11 } }, tickfont: { size: 9 } },
    zaxis: { title: { text: 'Intensity', font: { size: 11 } }, tickfont: { size: 9 } },
    camera: { eye: { x: 1.5, y: 1.6, z: 1.0 } }
  }
}))

usePlotly(el, plotData, layout)
</script>

<template>
  <div class="chart-wrap">
    <ChartActions :el="el" name="eem_surface" />
    <div ref="el" class="chart-box" :style="{ height: height || '420px' }"></div>
  </div>
</template>
