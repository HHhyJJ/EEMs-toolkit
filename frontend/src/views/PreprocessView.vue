<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import api, { errorMessage } from '../api'
import { useSessionStore } from '../stores/session'
import EemHeatmap from '../components/EemHeatmap.vue'
import EemSurface from '../components/EemSurface.vue'
import HistoryTimeline from '../components/HistoryTimeline.vue'

const { t } = useI18n()
const store = useSessionStore()

const sample = ref(1)
const compare = ref(false)
const use3d = ref(false)
const stride = ref(1)
const current = ref<any>(null)
const previous = ref<any>(null)
const busy = ref(false)

const ramanEx = ref(350)
const ramanRange = ref<[number, number]>([378, 424])
const exLt = ref<number | null>(null)
const exGt = ref<number | null>(null)
const emLt = ref<number | null>(null)
const emGt = ref<number | null>(null)
const dilutionText = ref('')
const rayBelow = ref(15)
const rayAbove = ref(15)
const ray2Below = ref(20)
const ray2Above = ref(15)
const ramBelow = ref(15)
const ramAbove = ref(15)
const ram2Below = ref(10)
const ram2Above = ref(10)
const sigma = ref(0.5)
const contourCount = ref(0)

const autoStride = computed(() => {
  const cells = (store.state.nex ?? 0) * (store.state.nem ?? 0)
  return cells > 24000 ? 2 : 1
})
const effectiveStride = computed(() => Math.max(stride.value, autoStride.value))

async function loadChart() {
  if (!store.imported) return
  if (sample.value > store.state.nSample) sample.value = 1
  const params = { sample: sample.value, stride: effectiveStride.value }
  current.value = (await api.get('/api/charts/eem', { params })).data
  if (compare.value) {
    previous.value = (await api.get('/api/charts/eem', { params: { ...params, state: 'previous' } })).data
  } else {
    previous.value = null
  }
}

async function apply(url: string, payload?: any) {
  busy.value = true
  try {
    await store.apply(url, payload)
    await loadChart()
    ElMessage.success(t('preprocess.applied'))
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    busy.value = false
  }
}

async function applySubset() {
  const payload: any = {}
  if (exLt.value !== null) payload.ex_lt = exLt.value
  if (exGt.value !== null) payload.ex_gt = exGt.value
  if (emLt.value !== null) payload.em_lt = emLt.value
  if (emGt.value !== null) payload.em_gt = emGt.value
  await apply('/api/preprocess/sub_dataset', payload)
}

async function applyDilute() {
  const text = dilutionText.value.trim()
  const factors = text ? text.split(/[,;\s]+/).filter(Boolean).map(Number) : undefined
  await apply('/api/preprocess/dilute', factors ? { factors } : {})
}

async function jump(index: number) {
  busy.value = true
  try {
    await store.jump(index)
    await loadChart()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    busy.value = false
  }
}

async function trimHistory(index: number) {
  busy.value = true
  try {
    await store.trim(index)
    await loadChart()
    ElMessage.success(t('preprocess.trimmed', { index }))
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    busy.value = false
  }
}

watch(() => store.state.revision, loadChart)
watch([sample, compare, effectiveStride], loadChart)
onMounted(loadChart)
</script>

<template>
  <div>
    <h2 class="page-title">{{ t('preprocess.title') }}</h2>
    <p class="page-hint">{{ t('preprocess.historyHint') }}</p>

    <el-alert
      v-if="store.state.facNums && store.state.facNums.length"
      type="warning"
      show-icon
      :closable="false"
      :title="t('preprocess.parafacWarning')"
      style="margin-bottom: 12px"
    />

    <el-alert
      v-if="store.state.nanCount"
      type="warning"
      show-icon
      :closable="false"
      :title="t('preprocess.nanWarning', { count: store.state.nanCount })"
      style="margin-bottom: 12px"
    />

    <el-row :gutter="12">
      <el-col :span="8">
        <el-card shadow="never" class="section-card">
          <template #header>{{ t('preprocess.steps') }}</template>

          <div class="step">
            <div class="step-head">{{ t('preprocess.innerFilter') }}<el-tag v-if="store.state.ife" size="small" type="success">{{ t('preprocess.completed') }}</el-tag></div>
            <el-button size="small" type="primary" :disabled="!store.state.hasAbs || busy || store.state.ife" @click="apply('/api/preprocess/inner_effect_correct')">
              {{ t('preprocess.apply') }}
            </el-button>
          </div>

          <el-divider />

          <div class="step">
            <div class="step-head">{{ t('preprocess.blankSubtraction') }}<el-tag v-if="store.state.blanked" size="small" type="success">{{ t('preprocess.completed') }}</el-tag></div>
            <el-button size="small" type="primary" :disabled="!store.state.hasBlank || busy || store.state.blanked" @click="apply('/api/preprocess/minus_blank')">
              {{ t('preprocess.apply') }}
            </el-button>
          </div>

          <el-divider />

          <div class="step-block">
            <div class="step-head">
              {{ t('preprocess.ramanNormalization') }}
              <el-tag v-if="store.state.unit === 'RU'" size="small" type="success">{{ t('preprocess.completed') }}</el-tag>
            </div>
            <div style="font-size: 12px; color: #909399; margin-bottom: 6px">
              {{ t('preprocess.ramanDescription') }}
            </div>
            <div class="form-inline">
              <span>{{ t('preprocess.ramanPeakEx') }}</span>
              <el-input-number v-model="ramanEx" :min="200" :max="600" size="small" controls-position="right" style="width: 100px" />
              <span>{{ t('preprocess.integrationEmRange') }}</span>
              <el-input-number v-model="ramanRange[0]" :min="300" :max="600" size="small" controls-position="right" style="width: 90px" />
              <span>-</span>
              <el-input-number v-model="ramanRange[1]" :min="300" :max="700" size="small" controls-position="right" style="width: 90px" />
            </div>
            <el-button size="small" type="primary" :disabled="!store.state.hasBlank || busy || store.state.unit === 'RU'" style="margin-top: 8px" @click="apply('/api/preprocess/raman_areal', { ex: ramanEx, em_range: ramanRange })">
              {{ t('preprocess.applyRamanNormalization') }}
            </el-button>
            <div v-if="!store.state.hasBlank" style="font-size: 12px; color: #e6a23c">
              {{ t('preprocess.blankRequired') }}
            </div>
            <div v-else-if="store.state.unit === 'RU'" style="font-size: 12px; color: #909399">
              {{ t('preprocess.ramanCompleted') }}
            </div>
          </div>

          <el-divider />

          <div class="step-block">
            <div class="step-head">{{ t('preprocess.trimWavelengths') }}</div>
            <div class="form-grid">
              <el-input-number v-model="exLt" :min="200" :max="700" size="small" placeholder="Ex <" controls-position="right" />
              <el-input-number v-model="exGt" :min="200" :max="700" size="small" placeholder="Ex >" controls-position="right" />
              <el-input-number v-model="emLt" :min="200" :max="800" size="small" placeholder="Em <" controls-position="right" />
              <el-input-number v-model="emGt" :min="200" :max="800" size="small" placeholder="Em >" controls-position="right" />
            </div>
            <el-button size="small" type="primary" :disabled="busy" style="margin-top: 8px" @click="applySubset">{{ t('preprocess.apply') }}</el-button>
          </div>

          <el-divider />

          <div class="step-block">
            <div class="step-head">{{ t('preprocess.dilutionCorrection') }}</div>
            <el-input v-model="dilutionText" size="small" :placeholder="t('preprocess.dilutionPlaceholder')" />
            <el-button size="small" type="primary" :disabled="busy" style="margin-top: 8px" @click="applyDilute">{{ t('preprocess.apply') }}</el-button>
          </div>

          <el-divider />

          <div class="step-block">
            <div class="step-head">{{ t('preprocess.scatterRemoval') }}</div>
            <div class="scatter-grid">
              <div class="scatter-group">
                <div class="scatter-title">{{ t('preprocess.rayleigh1') }}</div>
                <div class="form-inline">
                  <span>{{ t('preprocess.below') }}</span>
                  <el-input-number v-model="rayBelow" size="small" controls-position="right" style="width: 90px" />
                  <span>{{ t('preprocess.above') }}</span>
                  <el-input-number v-model="rayAbove" size="small" controls-position="right" style="width: 90px" />
                </div>
              </div>
              <div class="scatter-group">
                <div class="scatter-title">{{ t('preprocess.rayleigh2') }}</div>
                <div class="form-inline">
                  <span>{{ t('preprocess.below') }}</span>
                  <el-input-number v-model="ray2Below" size="small" controls-position="right" style="width: 90px" />
                  <span>{{ t('preprocess.above') }}</span>
                  <el-input-number v-model="ray2Above" size="small" controls-position="right" style="width: 90px" />
                </div>
              </div>
              <div class="scatter-group">
                <div class="scatter-title">{{ t('preprocess.raman1') }}</div>
                <div class="form-inline">
                  <span>{{ t('preprocess.below') }}</span>
                  <el-input-number v-model="ramBelow" size="small" controls-position="right" style="width: 90px" />
                  <span>{{ t('preprocess.above') }}</span>
                  <el-input-number v-model="ramAbove" size="small" controls-position="right" style="width: 90px" />
                </div>
              </div>
              <div class="scatter-group">
                <div class="scatter-title">{{ t('preprocess.raman2') }}</div>
                <div class="form-inline">
                  <span>{{ t('preprocess.below') }}</span>
                  <el-input-number v-model="ram2Below" size="small" controls-position="right" style="width: 90px" />
                  <span>{{ t('preprocess.above') }}</span>
                  <el-input-number v-model="ram2Above" size="small" controls-position="right" style="width: 90px" />
                </div>
              </div>
            </div>
            <el-button
              size="small"
              type="primary"
              :disabled="busy"
              style="margin-top: 8px"
              @click="apply('/api/preprocess/cut_scatter', { ray: [rayBelow, rayAbove], second_ray: [ray2Below, ray2Above], ram: [ramBelow, ramAbove], second_ram: [ram2Below, ram2Above] })"
            >
              {{ t('preprocess.apply') }}
            </el-button>
          </div>

          <el-divider />

          <div class="step-block">
            <div class="step-head">{{ t('preprocess.missingInterpolation') }}</div>
            <el-button size="small" type="primary" :disabled="busy" @click="apply('/api/preprocess/interpolate')">{{ t('preprocess.apply') }}</el-button>
          </div>

          <el-divider />

          <div class="step-block">
            <div class="step-head">{{ t('preprocess.gaussianSmoothing') }}</div>
            <div class="form-inline">
              <span>sigma</span>
              <el-input-number v-model="sigma" :min="0.1" :max="5" :step="0.1" size="small" controls-position="right" style="width: 110px" />
            </div>
            <el-button size="small" type="primary" :disabled="busy" style="margin-top: 8px" @click="apply('/api/preprocess/smooth', { sigma })">{{ t('preprocess.apply') }}</el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card shadow="never" class="section-card">
          <div class="toolbar">
            <span>{{ t('preprocess.sample') }}</span>
            <el-select v-model="sample" size="small" style="width: 190px" filterable>
              <el-option v-for="item in store.samples" :key="item.index" :label="`${item.index}. ${item.file}`" :value="item.index" />
            </el-select>
            <el-checkbox v-model="compare">{{ t('preprocess.comparePrevious') }}</el-checkbox>
            <el-checkbox v-model="use3d">{{ t('preprocess.surface3d') }}</el-checkbox>
            <span style="font-size: 12px; color: #909399">{{ t('preprocess.stride', { value: effectiveStride }) }}</span>
            <span>{{ t('preprocess.contours') }}</span>
            <el-input-number v-model="contourCount" :min="0" :max="80" size="small" controls-position="right" style="width: 100px" />
            <span style="font-size: 12px; color: #909399">{{ t('preprocess.autoContours') }}</span>
            <el-tag v-if="current" size="small" type="info">{{ current.title }} · {{ current.unit }}</el-tag>
          </div>

          <div v-if="!current" style="color: #909399">{{ t('preprocess.noData') }}</div>
          <div v-else class="heatmap-grid">
            <div v-if="compare">
              <el-card shadow="never" class="section-card">
                <template #header>{{ t('preprocess.previousStep', { index: store.history.position - 1 }) }}</template>
                <EemHeatmap v-if="previous" :ex="previous.ex" :em="previous.em" :z="previous.z" :zmin="previous.zmin" :zmax="previous.zmax" :contour-count="contourCount" />
              </el-card>
              <el-card shadow="never" class="section-card">
                <template #header>{{ t('preprocess.currentStep', { index: store.history.position }) }}</template>
                <EemHeatmap :ex="current.ex" :em="current.em" :z="current.z" :zmin="current.zmin" :zmax="current.zmax" :contour-count="contourCount" />
              </el-card>
            </div>
            <template v-else>
              <EemSurface v-if="use3d" :ex="current.ex" :em="current.em" :z="current.z" height="460px" />
              <EemHeatmap v-else :ex="current.ex" :em="current.em" :z="current.z" :zmin="current.zmin" :zmax="current.zmax" :contour-count="contourCount" :title="t('preprocess.sampleTitle', { order: current.sample.order, file: current.sample.file })" />
            </template>
          </div>
          <div v-if="current && current.nanCount" style="color: #e6a23c; font-size: 12px; margin-top: 8px">
            {{ t('preprocess.currentNan', { count: current.nanCount }) }}
          </div>
        </el-card>

        <el-card shadow="never">
          <template #header>{{ t('preprocess.historyTimeline') }}</template>
          <HistoryTimeline :entries="store.history.entries" :position="store.history.position" @jump="jump" @trim="trimHistory" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.step {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.step-head {
  font-weight: 600;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.step-block {
  margin-bottom: 4px;
}

.scatter-grid {
  display: grid;
  gap: 8px;
  grid-template-columns: 1fr;
}

.scatter-group {
  padding: 6px 8px;
  background: #fafafa;
  border-radius: 4px;
}

.scatter-title {
  font-size: 12px;
  font-weight: 600;
  color: #606266;
  margin-bottom: 4px;
}
</style>
