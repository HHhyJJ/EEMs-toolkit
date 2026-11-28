<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import api, { errorMessage } from '../api'
import { useSessionStore } from '../stores/session'
import JobPanel from '../components/JobPanel.vue'
import LineChart from '../components/LineChart.vue'
import HeatmapMultiples from '../components/HeatmapMultiples.vue'

const { t } = useI18n()
const store = useSessionStore()
const router = useRouter()

const fText = ref('2-4')
const starts = ref(10)
const maxIter = ref(2500)
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

const selectedF = ref<number | null>(null)
const residualSample = ref(1)

const loadings = ref<any>(null)
const fmax = ref<any>(null)
const core = ref<any>(null)
const sse = ref<any>(null)
const fingers = ref<any>(null)
const residual = ref<any>(null)

const exLoadingsSeries = computed(() => (loadings.value?.series ?? []).map((s: any, index: number) => ({ name: t('parafacModel.component', { n: index + 1 }), data: s.ex })))
const emLoadingsSeries = computed(() => (loadings.value?.series ?? []).map((s: any, index: number) => ({ name: t('parafacModel.component', { n: index + 1 }), data: s.em })))
const fmaxSeries = computed(() => (fmax.value?.series ?? []).map((s: any, index: number) => ({ name: t('parafacModel.component', { n: index + 1 }), data: s.values })))
const ccSeries = computed(() => (core.value ? [{ name: t('parafacModel.coreConsistency'), data: core.value.coreConsistency }] : []))
const erSeries = computed(() => (core.value ? [{ name: t('parafacModel.explanationPercent'), data: core.value.explanationPercent }] : []))
const sseTotalSeries = computed(() => (sse.value ? [{ name: 'SSE', data: sse.value.total }] : []))
const sseSampleSeries = computed(() => (sse.value?.samples?.series ?? []).map((item: any) => ({ name: `f=${item.f}`, data: item.values })))
const sseEmSeries = computed(() => (sse.value?.em?.series ?? []).map((item: any) => ({ name: `f=${item.f}`, data: item.values })))
const sseExSeries = computed(() => (sse.value?.ex?.series ?? []).map((item: any) => ({ name: `f=${item.f}`, data: item.values })))
const fingerItems = computed(() => {
  if (!fingers.value) return []
  return fingers.value.components.map((c: any, index: number) => ({
    title: t('parafacModel.component', { n: index + 1 }), ex: fingers.value.ex, em: fingers.value.em, z: c.z, zmin: c.min, zmax: c.max
  }))
})
const residualItems = computed(() => {
  if (!residual.value) return []
  return [
    { title: t('parafacModel.originalEem'), ex: residual.value.ex, em: residual.value.em, z: residual.value.sample.z, zmin: residual.value.sample.min, zmax: residual.value.sample.max },
    { title: t('parafacModel.modelTitle', { f: selectedF.value }), ex: residual.value.ex, em: residual.value.em, z: residual.value.model.z, zmin: residual.value.model.min, zmax: residual.value.model.max },
    { title: t('parafacModel.residual'), ex: residual.value.ex, em: residual.value.em, z: residual.value.residual.z, zmin: residual.value.residual.min, zmax: residual.value.residual.max }
  ]
})

async function startFit() {
  if (!fs.value.length) {
    ElMessage.warning(t('parafacModel.missingComponents'))
    return
  }
  try {
    await store.startJob('parafac', { f: fs.value, starts: starts.value, max_iter: maxIter.value })
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function stopFit() {
  try {
    await store.stopJob()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function loadF() {
  if (!selectedF.value) return
  const params = { f: selectedF.value }
  const [load, fm, c, fg] = await Promise.all([
    api.get('/api/charts/loadings', { params }),
    api.get('/api/charts/fmax', { params }),
    api.get('/api/charts/core'),
    api.get('/api/charts/fingers', { params })
  ])
  loadings.value = load.data
  fmax.value = fm.data
  core.value = c.data
  fingers.value = fg.data
  await loadResidual()
}

async function loadSse() {
  if (!store.state.facNums?.length) {
    sse.value = null
    return
  }
  const { data } = await api.get('/api/charts/sse')
  sse.value = data
}

async function loadResidual() {
  if (!selectedF.value) return
  if (residualSample.value > store.state.nSample) residualSample.value = 1
  const { data } = await api.get('/api/charts/residual', {
    params: { f: selectedF.value, sample: residualSample.value }
  })
  residual.value = data
}

watch(selectedF, loadF)
watch(residualSample, loadResidual)
watch(() => store.state.facNums, (nums: number[]) => {
  if (!nums || !nums.length) {
    selectedF.value = null
    sse.value = null
    return
  }
  loadSse()
  if (!selectedF.value || !nums.includes(selectedF.value)) selectedF.value = nums[0]
}, { deep: true, immediate: true })
watch(() => store.job?.state, (state) => {
  if (state === 'done') {
    const nums = store.state.facNums || []
    if (nums.length) selectedF.value = nums.includes(selectedF.value as number) ? selectedF.value : nums[0]
    loadSse()
    if (selectedF.value) loadF()
  }
})

onMounted(async () => {
  if (!store.state.imported) return
  if (store.state.facNums?.length) {
    selectedF.value = store.state.facNums[0]
    await loadF()
  }
})
</script>

<template>
  <div>
    <h2 class="page-title">{{ t('parafacModel.title') }}</h2>
    <p class="page-hint">
      {{ t('parafacModel.hint') }}
    </p>

    <el-alert
      type="info"
      show-icon
      :closable="false"
      :title="t('parafacModel.independenceNotice')"
      style="margin-bottom: 12px"
    />

    <el-card shadow="never" class="section-card">
      <div class="toolbar">
        <span>{{ t('parafacModel.componentCount') }}</span>
        <el-input v-model="fText" size="small" style="width: 180px" :placeholder="t('parafacModel.componentPlaceholder')" />
        <el-tag v-for="f in fs" :key="f" size="small" type="info">f={{ f }}</el-tag>
        <span>starts</span>
        <el-input-number v-model="starts" :min="1" :max="100" size="small" controls-position="right" style="width: 100px" />
        <span>{{ t('parafacModel.maxIterations') }}</span>
        <el-input-number v-model="maxIter" :min="10" :max="10000" :step="100" size="small" controls-position="right" style="width: 120px" />
        <el-button type="primary" :disabled="running || !store.imported" @click="startFit">{{ t('parafacModel.startFit') }}</el-button>
        <el-button type="danger" :disabled="!running" @click="stopFit">{{ t('parafacModel.stop') }}</el-button>
      </div>
      <div style="font-size: 12px; color: #909399">
        {{ t('parafacModel.fitHint') }}
      </div>
    </el-card>

    <JobPanel :job="store.job" />

    <template v-if="store.state.facNums && store.state.facNums.length">
      <el-card shadow="never" class="section-card">
        <div class="toolbar">
          <span>{{ t('parafacModel.viewComponentCount') }}</span>
          <el-radio-group v-model="selectedF">
            <el-radio-button v-for="n in store.state.facNums" :key="n" :value="n">{{ n }}</el-radio-button>
          </el-radio-group>
          <el-tag v-if="store.state.coreConsistency" type="info">
            CC {{ (store.state.coreConsistency[String(selectedF)] ?? 0).toFixed(2) }} ·
            {{ t('parafacModel.explanationRate') }} {{ ((store.state.explanationRate[String(selectedF)] ?? 0) * 100).toFixed(2) }}%
          </el-tag>
          <el-button size="small" @click="router.push('/parafac/outlier')">{{ t('parafacModel.outlierDetection') }}</el-button>
          <el-button size="small" @click="router.push('/parafac/validation')">{{ t('parafacModel.splitValidation') }}</el-button>
          <el-button size="small" type="success" @click="router.push('/parafac/result')">{{ t('parafacModel.resultsAndExport') }}</el-button>
        </div>
      </el-card>

      <el-row :gutter="12">
        <el-col :span="12">
          <el-card shadow="never" class="section-card">
            <template #header>{{ t('parafacModel.coreAndExplanation') }}</template>
            <el-row :gutter="8">
              <el-col :span="12">
                <LineChart :x="core?.facNums ?? []" :series="ccSeries" y-name="CC" height="260px" />
              </el-col>
              <el-col :span="12">
                <LineChart :x="core?.facNums ?? []" :series="erSeries" y-name="%" height="260px" />
              </el-col>
            </el-row>
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card shadow="never" class="section-card">
            <template #header>{{ t('parafacModel.fmaxTitle') }}</template>
            <LineChart :x="fmax?.labels ?? []" :series="fmaxSeries" y-name="FMax" height="260px" />
          </el-card>
        </el-col>
      </el-row>

      <el-card v-if="sse" shadow="never" class="section-card">
        <template #header>{{ t('parafacModel.sseTitle') }}</template>
        <div style="font-size: 12px; color: #909399; margin-bottom: 8px">{{ t('parafacModel.sseHint') }}</div>
        <el-row :gutter="8">
          <el-col :span="12">
            <LineChart :x="sse.facNums" :series="sseTotalSeries" :title="t('parafacModel.sseTotal')" y-name="SSE" height="280px" />
          </el-col>
          <el-col :span="12">
            <LineChart :x="sse.samples?.labels ?? []" :series="sseSampleSeries" :title="t('parafacModel.sseBySample')" y-name="SSE" height="280px" />
          </el-col>
        </el-row>
        <el-row :gutter="8" style="margin-top: 8px">
          <el-col :span="12">
            <LineChart :x="sse.em?.wavelengths ?? []" :series="sseEmSeries" :title="t('parafacModel.sseByEmission')" x-name="Em / nm" y-name="SSE" height="280px" />
          </el-col>
          <el-col :span="12">
            <LineChart :x="sse.ex?.wavelengths ?? []" :series="sseExSeries" :title="t('parafacModel.sseByExcitation')" x-name="Ex / nm" y-name="SSE" height="280px" />
          </el-col>
        </el-row>
      </el-card>

      <el-card shadow="never" class="section-card">
        <template #header>{{ t('parafacModel.loadingsTitle') }}</template>
        <el-row :gutter="8">
          <el-col :span="12">
            <LineChart :x="loadings?.ex ?? []" :series="exLoadingsSeries" x-name="Ex / nm" y-name="Loading" height="280px" />
          </el-col>
          <el-col :span="12">
            <LineChart :x="loadings?.em ?? []" :series="emLoadingsSeries" x-name="Em / nm" y-name="Loading" height="280px" />
          </el-col>
        </el-row>
      </el-card>

      <el-card shadow="never" class="section-card">
        <template #header>{{ t('parafacModel.componentPlotsTitle') }}</template>
        <HeatmapMultiples :items="fingerItems" />
        <div v-if="fingers" style="margin-top: 10px; font-size: 12px; color: #606266">
          <div v-for="(peak, index) in fingers.peaks" :key="peak.component">
            <b>{{ t('parafacModel.component', { n: index + 1 }) }}</b> — {{ t('parafacModel.peakValues', { em: peak.em.join(', ') || '—', ex: peak.ex.join(', ') || '—' }) }}
          </div>
        </div>
      </el-card>

      <el-card shadow="never">
        <template #header>
          <div style="display: flex; justify-content: space-between; align-items: center">
            <span>{{ t('parafacModel.residualView') }}</span>
            <el-select v-model="residualSample" size="small" style="width: 190px" filterable>
              <el-option v-for="item in store.samples" :key="item.index" :label="`${item.index}. ${item.file}`" :value="item.index" />
            </el-select>
          </div>
        </template>
        <HeatmapMultiples :items="residualItems" />
      </el-card>
    </template>
    <el-empty v-else-if="!running" :description="t('parafacModel.empty')" />
  </div>
</template>
