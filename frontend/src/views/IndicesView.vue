<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import api, { downloadUrl, errorMessage } from '../api'
import { useSessionStore } from '../stores/session'
import LineChart from '../components/LineChart.vue'
import EemHeatmap from '../components/EemHeatmap.vue'

const { t } = useI18n()
const store = useSessionStore()
const tab = ref('indices')
const indices = ref<any>(null)
const fri = ref<any>(null)
const slope = ref<any>(null)
const friChart = ref<any>(null)
const friSample = ref(1)
const friContours = ref(0)
const indexColumn = ref('')
const loading = ref(false)

function friColumnLabel(column: string) {
  const regions = [
    t('indices.region1'),
    t('indices.region2'),
    t('indices.region3'),
    t('indices.region4'),
    t('indices.region5')
  ]
  return regions[Number(column) - 1] ?? column
}

const indexColumns = computed(() => (indices.value?.columns ?? []).filter((c: string) => c !== 'Sample'))
const indexSeries = computed(() => {
  if (!indices.value || !indexColumn.value) return []
  return [{
    name: indexColumn.value,
    data: indices.value.rows.map((row: any) => row[indexColumn.value]),
    type: 'bar'
  }]
})
const indexLabels = computed(() => (indices.value?.rows ?? []).map((row: any) => String(row.Sample)))

const friZ = computed(() => {
  const z = friChart.value?.z
  if (!z || !z.length) return []
  return z[0].map((_: any, xi: number) => z.map((row: any) => row[xi]))
})

const friRegions = computed(() => {
  const chart = friChart.value
  if (!chart) return undefined
  const emMin = chart.em[0]
  const emMax = chart.em[chart.em.length - 1]
  const exMin = chart.ex[0]
  const exMax = chart.ex[chart.ex.length - 1]
  const labels = [
    { x: (emMin + 330) / 2, y: (exMin + 250) / 2, text: 'Ⅰ' },
    { x: (330 + 380) / 2, y: (exMin + 250) / 2, text: 'Ⅱ' },
    { x: (380 + emMax) / 2, y: (exMin + 250) / 2, text: 'Ⅲ' },
    { x: (emMin + 380) / 2, y: (exMax + 250) / 2, text: 'Ⅳ' },
    { x: (380 + emMax) / 2, y: (exMax + 250) / 2, text: 'Ⅴ' }
  ].filter((label) => label.x >= emMin && label.x <= emMax && label.y >= exMin && label.y <= exMax)
  return {
    labels,
    lines: [
      { x1: emMin, y1: 250, x2: emMax, y2: 250 },
      { x1: 380, y1: exMin, x2: 380, y2: exMax },
      { x1: 330, y1: exMin, x2: 330, y2: 250 }
    ]
  }
})

async function loadIndices() {
  if (indices.value) return
  loading.value = true
  try {
    indices.value = (await api.get('/api/indices')).data
    indexColumn.value = indexColumns.value[0] ?? ''
  } catch (error) {
    ElMessage.error(t('indices.loadIndicesFailed', { message: errorMessage(error) }))
  } finally {
    loading.value = false
  }
}

async function loadFri() {
  if (fri.value) {
    await loadFriChart()
    return
  }
  if (!store.state.friAvailable) return
  loading.value = true
  try {
    fri.value = (await api.get('/api/fri')).data
    await loadFriChart()
  } catch (error) {
    ElMessage.error(t('indices.loadFriFailed', { message: errorMessage(error) }))
  } finally {
    loading.value = false
  }
}

async function loadFriChart() {
  const { data } = await api.get('/api/charts/fri', { params: { sample: friSample.value } })
  friChart.value = data
}

async function loadSlope() {
  if (slope.value) return
  loading.value = true
  try {
    slope.value = (await api.get('/api/slope')).data
  } catch (error) {
    ElMessage.error(t('indices.loadSlopeFailed', { message: errorMessage(error) }))
  } finally {
    loading.value = false
  }
}

function loadCurrent() {
  if (tab.value === 'indices') loadIndices()
  else if (tab.value === 'fri') loadFri()
  else loadSlope()
}

watch(tab, loadCurrent)
watch(friSample, () => {
  if (tab.value === 'fri') loadFriChart()
})
onMounted(loadCurrent)
</script>

<template>
  <div>
    <h2 class="page-title">{{ t('indices.pageTitle') }}</h2>
    <p class="page-hint">
      {{ t('indices.pageHint') }}
    </p>

    <el-tabs v-model="tab" type="border-card">
      <el-tab-pane :label="t('indices.fluorescenceIndices')" name="indices">
        <div v-loading="loading">
          <div class="toolbar">
            <span>{{ t('indices.chartMetric') }}</span>
            <el-select v-model="indexColumn" size="small" style="width: 160px">
              <el-option v-for="col in indexColumns" :key="col" :label="col" :value="col" />
            </el-select>
            <el-button size="small" @click="downloadUrl('/api/export/indices', 'fluorescence_indices.xlsx')">{{ t('indices.downloadExcel') }}</el-button>
          </div>
          <LineChart v-if="indexColumn" :x="indexLabels" :series="indexSeries" y-name="Value" height="300px" :legend="false" />
          <div class="table-scroll" style="margin-top: 12px">
            <el-table v-if="indices" :data="indices.rows" size="small" border height="420" style="width: 100%">
              <el-table-column v-for="col in indices.columns" :key="col" :prop="col" :label="col" sortable min-width="100" />
            </el-table>
          </div>
        </div>
      </el-tab-pane>

      <el-tab-pane label="FRI" name="fri">
        <el-alert
          v-if="!store.state.friAvailable"
          type="warning"
          show-icon
          :closable="false"
          :title="t('indices.friUnavailable')"
          style="margin-bottom: 12px"
        />
        <div v-else v-loading="loading">
          <div class="toolbar">
            <span>{{ t('indices.sample') }}</span>
            <el-select v-model="friSample" size="small" style="width: 190px" filterable>
              <el-option v-for="item in store.samples" :key="item.index" :label="`${item.index}. ${item.file}`" :value="item.index" />
            </el-select>
            <span>{{ t('indices.contours') }}</span>
            <el-input-number v-model="friContours" :min="0" :max="80" size="small" controls-position="right" style="width: 100px" />
            <span style="font-size: 12px; color: #909399">{{ t('indices.auto') }}</span>
            <el-button size="small" @click="downloadUrl('/api/export/fri', 'FRI.xlsx')">{{ t('indices.downloadExcel') }}</el-button>
          </div>
          <el-row v-if="friChart" :gutter="12">
            <el-col :span="14">
              <EemHeatmap
                :ex="friChart.em"
                :em="friChart.ex"
                :z="friZ"
                :zmin="friChart.zmin"
                :zmax="friChart.zmax"
                :regions="friRegions"
                :contour-count="friContours"
                x-name="Em / nm"
                y-name="Ex / nm"
                :title="t('indices.chartTitle', { order: friChart.sample.order, file: friChart.sample.file })"
              />
            </el-col>
            <el-col :span="10">
              <div v-for="table in fri?.tables ?? []" :key="table.name" style="margin-bottom: 10px">
                <div style="font-size: 12px; font-weight: 600; margin-bottom: 4px">{{ table.name }}</div>
                <el-table :data="table.rows" size="small" border>
                  <el-table-column type="index" label="#" width="46" />
                  <el-table-column v-for="col in table.columns" :key="col" :prop="col" :label="friColumnLabel(col)" min-width="72" />
                </el-table>
              </div>
            </el-col>
          </el-row>
          <div v-else style="color: #909399">{{ t('indices.noFriResults') }}</div>
        </div>
      </el-tab-pane>

      <el-tab-pane :label="t('indices.spectralSlope')" name="slope">
        <div v-loading="loading">
          <el-alert
            v-if="!store.state.hasAbs"
            type="warning"
            show-icon
            :closable="false"
            :title="t('indices.slopeUnavailable')"
            style="margin-bottom: 12px"
          />
          <div v-else>
            <div class="toolbar">
              <el-button size="small" @click="downloadUrl('/api/export/slope', 'slope_fit_result.xlsx')">{{ t('indices.downloadExcel') }}</el-button>
            </div>
            <div class="table-scroll">
              <el-table v-if="slope" :data="slope.rows" size="small" border height="460" style="width: 100%">
                <el-table-column v-for="col in slope.columns" :key="col" :prop="col" :label="col" sortable min-width="130" />
              </el-table>
            </div>
          </div>
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>
