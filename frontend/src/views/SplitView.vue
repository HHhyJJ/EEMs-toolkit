<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import api, { errorMessage } from '../api'
import { useSessionStore } from '../stores/session'
import JobPanel from '../components/JobPanel.vue'
import LineChart from '../components/LineChart.vue'

const { t } = useI18n()
const store = useSessionStore()

const fText = ref('2-4')
const starts = ref(10)
const maxIter = ref(2500)
const useSeed = ref(false)
const seed = ref(0)
const results = ref<Record<number, any>>({})
const starting = ref(false)
const running = computed(() => store.job?.state === 'running')

function parseComponentNumbers(text: string): number[] {
  const out = new Set<number>()
  for (const part of text.split(/[,;，、\s]+/).filter(Boolean)) {
    const range = part.match(/^(\d+)\s*[-~]\s*(\d+)$/)
    if (range) {
      const from = Number(range[1])
      const to = Number(range[2])
      for (let i = Math.min(from, to); i <= Math.max(from, to); i++) out.add(i)
    } else if (/^\d+$/.test(part)) {
      out.add(Number(part))
    }
  }
  return [...out].filter((n) => n >= 1).sort((a, b) => a - b)
}

const fs = computed(() => parseComponentNumbers(fText.value))

async function startSplit() {
  if (!fs.value.length) {
    ElMessage.warning(t('split.validateComponents'))
    return
  }
  starting.value = true
  try {
    await store.startJob('split', {
      f: fs.value,
      starts: starts.value,
      max_iter: maxIter.value,
      random_state: useSeed.value ? seed.value : null
    })
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    starting.value = false
  }
}

async function stopSplit() {
  try {
    await store.stopJob()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function loadFor(numbers: number[]) {
  const loaded: Record<number, any> = {}
  for (const f of numbers) {
    try {
      const { data } = await api.get('/api/charts/split', { params: { f } })
      loaded[f] = data
    } catch {
      // Skip component numbers without a split result.
    }
  }
  results.value = loaded
}

function emSeries(comparison: any) {
  return comparison.em.flatMap((series: any, index: number) => [
    { name: `${t('split.component', { n: index + 1 })} 1st`, data: series.first },
    { name: `${t('split.component', { n: index + 1 })} 2nd`, data: series.second, dashed: true },
    { name: `${t('split.component', { n: index + 1 })} overall`, data: series.overall, dashed: true }
  ])
}

function exSeries(comparison: any) {
  return comparison.ex.flatMap((series: any, index: number) => [
    { name: `${t('split.component', { n: index + 1 })} 1st`, data: series.first },
    { name: `${t('split.component', { n: index + 1 })} 2nd`, data: series.second, dashed: true },
    { name: `${t('split.component', { n: index + 1 })} overall`, data: series.overall, dashed: true }
  ])
}

watch(() => store.job?.state, async (state) => {
  if (state === 'done') await loadFor(fs.value)
})
watch(() => store.state.hasSplit, async (has: boolean) => {
  if (has) await loadFor(store.state.facNums || [])
})

onMounted(async () => {
  if (!store.state.imported) return
  if (store.state.hasSplit) await loadFor(store.state.facNums || [])
})
</script>

<template>
  <div>
    <h2 class="page-title">{{ t('split.title') }}</h2>
    <p class="page-hint">{{ t('split.hint') }}</p>

    <el-card shadow="never" class="section-card">
      <div class="toolbar">
        <span>{{ t('split.componentCount') }}</span>
        <el-input v-model="fText" size="small" style="width: 180px" :placeholder="t('split.componentPlaceholder')" />
        <el-tag v-for="f in fs" :key="f" size="small" type="info">f={{ f }}</el-tag>
        <span>starts</span>
        <el-input-number v-model="starts" :min="1" :max="100" size="small" controls-position="right" style="width: 100px" />
        <span>{{ t('split.maxIterations') }}</span>
        <el-input-number v-model="maxIter" :min="10" :max="10000" :step="100" size="small" controls-position="right" style="width: 120px" />
        <el-checkbox v-model="useSeed">{{ t('split.fixedRandomState') }}</el-checkbox>
        <el-input-number v-if="useSeed" v-model="seed" :min="0" :max="99999" size="small" controls-position="right" style="width: 110px" />
        <el-button type="primary" :disabled="running || starting" @click="startSplit">{{ t('split.startSplit') }}</el-button>
        <el-button type="danger" :disabled="!running" @click="stopSplit">{{ t('split.stop') }}</el-button>
      </div>
      <div style="font-size: 12px; color: #909399">
        {{ t('split.randomStateHint') }}
      </div>
    </el-card>

    <JobPanel :job="store.job" />

    <template v-if="Object.keys(results).length">
      <el-card v-for="(split, key) in results" :key="key" shadow="never" class="section-card">
        <template #header>
          <div class="toolbar" style="margin: 0">
            <span>{{ t('split.tccValidation', { f: split.f }) }}</span>
            <el-tag :type="split.tcc.validated ? 'success' : 'danger'">
              {{ t('split.minimumTcc') }} {{ split.tcc.minTcc.toFixed(4) }} · {{ split.tcc.validated ? t('split.passed') : t('split.failed') }}
            </el-tag>
            <span style="font-size: 12px; color: #909399">{{ t('split.comparisonCount', { count: split.comparisons.length }) }}</span>
          </div>
        </template>
        <el-row :gutter="12">
          <el-col v-for="comparison in split.comparisons" :key="comparison.title" :span="24">
            <div style="font-size: 13px; font-weight: 600; margin: 6px 0">{{ comparison.title }}</div>
            <el-row :gutter="8">
              <el-col :span="12">
                <LineChart :x="split.em" :series="emSeries(comparison)" x-name="Em / nm" y-name="Loading" height="260px" />
              </el-col>
              <el-col :span="12">
                <LineChart :x="split.ex" :series="exSeries(comparison)" x-name="Ex / nm" y-name="Loading" height="260px" />
              </el-col>
            </el-row>
          </el-col>
        </el-row>
      </el-card>
    </template>
    <el-empty v-else-if="!running" :description="t('split.empty')" />
  </div>
</template>
