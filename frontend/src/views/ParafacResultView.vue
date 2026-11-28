<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import api, { downloadUrl, errorMessage } from '../api'
import { useSessionStore } from '../stores/session'
import LineChart from '../components/LineChart.vue'
import HeatmapMultiples from '../components/HeatmapMultiples.vue'

const { t } = useI18n()
const store = useSessionStore()
const router = useRouter()

const selectedF = ref<number | null>(null)
const loadings = ref<any>(null)
const fmax = ref<any>(null)
const fingers = ref<any>(null)

const exSeries = computed(() => (loadings.value?.series ?? []).map((s: any, index: number) => ({ name: t('parafacResult.component', { n: index + 1 }), data: s.ex })))
const emSeries = computed(() => (loadings.value?.series ?? []).map((s: any, index: number) => ({ name: t('parafacResult.component', { n: index + 1 }), data: s.em })))
const fmaxSeries = computed(() => (fmax.value?.series ?? []).map((s: any, index: number) => ({ name: t('parafacResult.component', { n: index + 1 }), data: s.values })))
const fingerItems = computed(() => {
  if (!fingers.value) return []
  return fingers.value.components.map((c: any, index: number) => ({
    title: t('parafacResult.component', { n: index + 1 }), ex: fingers.value.ex, em: fingers.value.em, z: c.z, zmin: c.min, zmax: c.max
  }))
})
const cc = computed(() => selectedF.value ? Number(store.state.coreConsistency?.[String(selectedF.value)] ?? 0) : 0)
const er = computed(() => selectedF.value ? Number(store.state.explanationRate?.[String(selectedF.value)] ?? 0) : 0)

async function load() {
  if (!selectedF.value) return
  try {
    const params = { f: selectedF.value }
    const [l, fm, fg] = await Promise.all([
      api.get('/api/charts/loadings', { params }),
      api.get('/api/charts/fmax', { params }),
      api.get('/api/charts/fingers', { params })
    ])
    loadings.value = l.data
    fmax.value = fm.data
    fingers.value = fg.data
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

watch(selectedF, load)
watch(() => store.state.facNums, (nums: number[]) => {
  if (!nums || !nums.length) {
    selectedF.value = null
    return
  }
  if (!selectedF.value || !nums.includes(selectedF.value)) selectedF.value = nums[0]
}, { deep: true, immediate: true })

onMounted(async () => {
  if (store.state.facNums?.length) await load()
})
</script>

<template>
  <div>
    <h2 class="page-title">{{ t('parafacResult.title') }}</h2>
    <p class="page-hint">{{ t('parafacResult.hint') }}</p>

    <template v-if="store.state.facNums && store.state.facNums.length">
      <el-card shadow="never" class="section-card">
        <div class="toolbar">
          <span>{{ t('parafacResult.componentCount') }}</span>
          <el-select v-model="selectedF" size="small" style="width: 130px">
            <el-option v-for="n in store.state.facNums" :key="n" :label="`f = ${n}`" :value="n" />
          </el-select>
          <el-button size="small" @click="router.push('/parafac/validation')">{{ t('parafacResult.goToValidation') }}</el-button>
          <el-button size="small" :disabled="!selectedF" @click="downloadUrl(`/api/export/openfluor?f=${selectedF}`, 'model.txt')">
            {{ t('parafacResult.downloadOpenfluor') }}
          </el-button>
          <el-button size="small" :disabled="!selectedF" @click="downloadUrl(`/api/export/parafac?f=${selectedF}`, 'parafac-result.xlsx')">
            {{ t('parafacResult.downloadExcel') }}
          </el-button>
        </div>
        <el-descriptions v-if="selectedF" :column="4" border size="small">
          <el-descriptions-item :label="t('parafacResult.componentCount')">{{ selectedF }}</el-descriptions-item>
          <el-descriptions-item :label="t('parafacResult.coreConsistencyCc')">{{ cc.toFixed(2) }}</el-descriptions-item>
          <el-descriptions-item :label="t('parafacResult.explanationRate')">{{ (er * 100).toFixed(2) }}%</el-descriptions-item>
          <el-descriptions-item :label="t('parafacResult.splitValidation')">
            <el-tag v-if="store.state.hasSplit" size="small" type="success">{{ t('parafacResult.ran') }}</el-tag>
            <el-tag v-else size="small" type="info">{{ t('parafacResult.notRun') }}</el-tag>
          </el-descriptions-item>
        </el-descriptions>
      </el-card>

      <el-row :gutter="12">
        <el-col :span="24">
          <el-card shadow="never" class="section-card">
            <template #header>FMax</template>
            <LineChart :x="fmax?.labels ?? []" :series="fmaxSeries" y-name="FMax" height="300px" />
          </el-card>
        </el-col>
      </el-row>

      <el-card shadow="never" class="section-card">
        <template #header>{{ t('parafacResult.loadingsTitle') }}</template>
        <el-row :gutter="8">
          <el-col :span="12">
            <LineChart :x="loadings?.ex ?? []" :series="exSeries" x-name="Ex / nm" y-name="Loading" height="280px" />
          </el-col>
          <el-col :span="12">
            <LineChart :x="loadings?.em ?? []" :series="emSeries" x-name="Em / nm" y-name="Loading" height="280px" />
          </el-col>
        </el-row>
      </el-card>

      <el-card shadow="never">
        <template #header>{{ t('parafacResult.componentPlotsTitle') }}</template>
        <HeatmapMultiples :items="fingerItems" />
      </el-card>
    </template>
    <el-empty v-else :description="t('parafacResult.empty')">
      <el-button type="primary" @click="router.push('/parafac/model')">{{ t('parafacResult.goToModel') }}</el-button>
    </el-empty>
  </div>
</template>
