<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import api, { downloadPost, downloadUrl, errorMessage } from '../api'
import FilePicker from '../components/FilePicker.vue'
import { useSessionStore } from '../stores/session'

const store = useSessionStore()
const { t } = useI18n()

const selectedF = ref<number | null>(null)
const includeHistory = ref(true)
const include = ref<string[]>(['openfluor', 'parafac', 'eems', 'indices', 'fri', 'slope'])
const busy = ref(false)

watch(() => store.state.facNums, (nums: number[] | undefined) => {
  const list = nums ?? []
  if (!list.length) {
    selectedF.value = null
    return
  }
  if (!selectedF.value || !list.includes(selectedF.value)) selectedF.value = list[0]
}, { deep: true, immediate: true })

const options = [
  { value: 'openfluor', labelKey: 'exportPage.optionOpenfluor', url: () => `/api/export/openfluor?f=${selectedF.value}`, filename: 'model.txt' },
  { value: 'parafac', labelKey: 'exportPage.optionParafac', url: () => `/api/export/parafac?f=${selectedF.value}`, filename: 'parafac-result.xlsx' },
  { value: 'eems', labelKey: 'exportPage.optionEems', url: () => '/api/export/eems', filename: 'eems.zip' },
  { value: 'indices', labelKey: 'exportPage.optionIndices', url: () => '/api/export/indices', filename: 'fluorescence_indices.xlsx' },
  { value: 'fri', labelKey: 'exportPage.optionFri', url: () => '/api/export/fri', filename: 'FRI.xlsx' },
  { value: 'slope', labelKey: 'exportPage.optionSlope', url: () => '/api/export/slope', filename: 'slope_fit_result.xlsx' }
]

const canExportF = computed(() => (store.state.facNums?.length ?? 0) > 0)

function needsComponent(item: (typeof options)[number]): boolean {
  return item.value === 'openfluor' || item.value === 'parafac'
}

function downloadOne(item: (typeof options)[number]) {
  if (needsComponent(item) && !selectedF.value) {
    ElMessage.warning(t('exportPage.selectComponentCount'))
    return
  }
  downloadUrl(item.url(), item.filename)
}

function downloadAll() {
  const params = new URLSearchParams()
  if (selectedF.value) params.set('f', String(selectedF.value))
  if (include.value.length) params.set('include', include.value.join(','))
  downloadUrl(`/api/export/all?${params.toString()}`, 'eems-export.zip')
}

async function saveProject() {
  busy.value = true
  try {
    await downloadPost('/api/project/save', { include_history: includeHistory.value, f: selectedF.value }, 'eems-project.eemproj')
    ElMessage.success(t('exportPage.projectExported'))
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    busy.value = false
  }
}

async function loadProject(file: File | null) {
  if (!file) return
  const form = new FormData()
  form.append('file', file)
  busy.value = true
  try {
    const { data } = await api.post('/api/project/load', form, { headers: { 'Content-Type': 'multipart/form-data' } })
    store.applyResult(data)
    await store.bootstrap()
    selectedF.value = store.state.selectedF ?? (store.state.facNums?.[0] ?? null)
    ElMessage.success(t('exportPage.projectLoaded'))
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div>
    <h2 class="page-title">{{ t('exportPage.title') }}</h2>
    <p class="page-hint">{{ t('exportPage.hint') }}</p>

    <el-card shadow="never" class="section-card">
      <template #header>{{ t('exportPage.optionsTitle') }}</template>
      <div class="toolbar">
        <span>{{ t('exportPage.componentCount') }}</span>
        <el-select v-model="selectedF" size="small" style="width: 130px" :placeholder="t('exportPage.selectF')" :disabled="!canExportF">
          <el-option v-for="n in store.state.facNums || []" :key="n" :label="`f = ${n}`" :value="n" />
        </el-select>
        <el-checkbox-group v-model="include">
          <el-checkbox v-for="item in options" :key="item.value" :value="item.value">{{ t(item.labelKey) }}</el-checkbox>
        </el-checkbox-group>
      </div>
      <div class="toolbar">
        <el-button type="primary" @click="downloadAll">{{ t('exportPage.downloadAll') }}</el-button>
        <el-button v-for="item in options" :key="item.value" size="small" @click="downloadOne(item)">
          {{ t('exportPage.download', { item: t(item.labelKey) }) }}
        </el-button>
      </div>
    </el-card>

    <el-card shadow="never">
      <template #header>{{ t('exportPage.projectFile') }}</template>
      <div class="toolbar">
        <el-checkbox v-model="includeHistory">{{ t('exportPage.includeHistory') }}</el-checkbox>
        <el-button type="primary" :loading="busy" @click="saveProject">{{ t('exportPage.saveProject') }}</el-button>
        <span style="font-size: 12px; color: #909399">{{ t('exportPage.saveDescription') }}</span>
      </div>
      <el-divider />
      <div class="toolbar">
        <span>{{ t('exportPage.loadProject') }}</span>
        <FilePicker accept=".eemproj,.zip" @change="loadProject($event[0] ?? null)" />
      </div>
    </el-card>
  </div>
</template>
