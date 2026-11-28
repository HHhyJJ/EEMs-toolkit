<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { saveChart, type ChartFormat } from '../charts/export'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps<{
  el: HTMLElement | null
  name: string
}>()

const format = ref<ChartFormat>('png')
const pixelRatio = ref(2)
const size = ref<{ width: number; height: number } | null>(null)
let observer: ResizeObserver | null = null

function measure() {
  if (!props.el) return
  const rect = props.el.getBoundingClientRect()
  if (rect.width && rect.height) {
    size.value = { width: Math.round(rect.width), height: Math.round(rect.height) }
  }
}

const resolutions = computed(() =>
  [1, 2, 3].map((ratio) => ({
    value: ratio,
    label: size.value ? `${size.value.width * ratio} × ${size.value.height * ratio}` : `${ratio}×`
  }))
)

function save() {
  saveChart(props.el, props.name, format.value, pixelRatio.value)
}

onMounted(() => {
  measure()
  if (props.el) {
    observer = new ResizeObserver(measure)
    observer.observe(props.el)
  }
})

watch(() => props.el, (node) => {
  observer?.disconnect()
  observer = null
  if (node) {
    observer = new ResizeObserver(measure)
    observer.observe(node)
  }
  measure()
})

onBeforeUnmount(() => {
  observer?.disconnect()
  observer = null
})
</script>

<template>
  <div class="chart-actions">
    <div class="chart-actions-extra">
      <slot />
    </div>
    <el-select v-model="format" size="small" style="width: 76px">
      <el-option label="PNG" value="png" />
      <el-option label="JPEG" value="jpeg" />
      <el-option label="SVG" value="svg" />
    </el-select>
    <template v-if="format !== 'svg'">
      <span class="chart-actions-label">{{ t('charts.resolution') }}</span>
      <el-select v-model="pixelRatio" size="small" style="width: 124px">
        <el-option v-for="item in resolutions" :key="item.value" :value="item.value" :label="item.label" />
      </el-select>
    </template>
    <el-button size="small" @click="save">{{ t('charts.saveImage') }}</el-button>
  </div>
</template>

<style>
.chart-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
  padding: 0 2px 6px;
}

.chart-actions-extra {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-right: auto;
}

.chart-actions-label {
  font-size: 12px;
  color: #909399;
}

.chart-wrap {
  position: relative;
  width: 100%;
  display: flex;
  flex-direction: column;
}

.chart-wrap .chart-box {
  flex: 1 1 auto;
  min-height: 0;
  width: 100%;
}

.chart-wrap.square {
  aspect-ratio: 1 / 1;
  width: 100%;
  max-width: 620px;
  min-height: 320px;
  max-height: 620px;
  margin: 0 auto;
}
</style>
