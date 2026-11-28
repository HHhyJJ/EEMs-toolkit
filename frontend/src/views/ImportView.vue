<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import api, { downloadUrl, errorMessage } from '../api'
import FilePicker from '../components/FilePicker.vue'
import { useSessionStore } from '../stores/session'

const store = useSessionStore()
const router = useRouter()
const { t } = useI18n()

const eemFiles = ref<File[]>([])
const absFiles = ref<File[]>([])
const blankFiles = ref<File[]>([])
const sampleLog = ref<File | null>(null)
const zipFile = ref<File | null>(null)
const busy = ref(false)
const error = ref('')

async function importUpload() {
  error.value = ''
  if (!zipFile.value && eemFiles.value.length === 0) {
    error.value = t('importData.selectEemOrZip')
    return
  }
  const form = new FormData()
  eemFiles.value.forEach((file) => form.append('eem_files', file))
  absFiles.value.forEach((file) => form.append('abs_files', file))
  blankFiles.value.forEach((file) => form.append('blank_files', file))
  if (sampleLog.value) form.append('sample_log', sampleLog.value)
  if (zipFile.value) form.append('zip_file', zipFile.value)
  busy.value = true
  try {
    const { data } = await api.post('/api/import', form, { headers: { 'Content-Type': 'multipart/form-data' } })
    store.applyResult(data)
    await store.bootstrap()
    ElMessage.success(t('importData.imported', { n: store.state.nSample }))
    router.push('/preprocess')
  } catch (err) {
    error.value = errorMessage(err)
  } finally {
    busy.value = false
  }
}

async function resumeProject(file: File | null) {
  if (!file) return
  const form = new FormData()
  form.append('file', file)
  busy.value = true
  try {
    const { data } = await api.post('/api/project/load', form, { headers: { 'Content-Type': 'multipart/form-data' } })
    store.applyResult(data)
    await store.bootstrap()
    ElMessage.success(t('importData.projectLoaded'))
    router.push('/preprocess')
  } catch (err) {
    error.value = errorMessage(err)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div>
    <h2 class="page-title">{{ t('importData.title') }}</h2>
    <p class="page-hint">{{ t('importData.hint') }}</p>

    <el-alert v-if="error" :title="error" type="error" show-icon closable style="margin-bottom: 12px" @close="error = ''" />

    <el-card shadow="never" class="section-card">
      <div class="file-rows">
        <div class="file-row">
          <span class="file-label">{{ t('importData.eemFiles') }}</span>
          <FilePicker multiple accept=".xlsx,.xls" @change="eemFiles = $event" />
        </div>
        <div class="file-row">
          <span class="file-label">{{ t('importData.absFiles') }}</span>
          <FilePicker multiple accept=".xlsx,.xls" @change="absFiles = $event" />
        </div>
        <div class="file-row">
          <span class="file-label">{{ t('importData.blankFiles') }}</span>
          <FilePicker multiple accept=".xlsx,.xls" @change="blankFiles = $event" />
        </div>
        <div class="file-row">
          <span class="file-label">{{ t('importData.sampleLog') }}</span>
          <FilePicker accept=".xlsx,.xls" @change="sampleLog = $event[0] ?? null" />
        </div>
        <div class="file-row">
          <span class="file-label">{{ t('importData.zipFile') }}</span>
          <FilePicker accept=".zip" @change="zipFile = $event[0] ?? null" />
        </div>
      </div>
      <div class="toolbar" style="margin-top: 14px">
        <el-button type="primary" :loading="busy" @click="importUpload">{{ t('importData.uploadImport') }}</el-button>
        <el-button @click="downloadUrl('/api/import/demo-package', 'eems-demo.zip')">{{ t('importData.downloadDemo') }}</el-button>
      </div>
    </el-card>

    <el-card shadow="never" class="section-card">
      <template #header>{{ t('importData.zipStructure') }}</template>
      <pre class="mono">eems-demo.zip
├─ eem/           {{ t('importData.zipEem') }}
│   ├─ 423SF.xlsx
│   └─ ...
├─ abs/           {{ t('importData.zipAbs') }}
│   └─ 423SF.xlsx
├─ blank/         {{ t('importData.zipBlank') }}
│   └─ di25se06.xlsx
└─ SampleLog.xlsx {{ t('importData.zipSampleLog') }}</pre>
      <ul style="font-size: 13px; color: #606266; line-height: 1.8; margin: 8px 0 0; padding-left: 18px">
        <li>{{ t('importData.withSampleLog') }}</li>
        <li>{{ t('importData.withoutSampleLog') }}</li>
        <li>{{ t('importData.demoNote') }}</li>
      </ul>
    </el-card>

    <el-card shadow="never" class="section-card">
      <template #header>{{ t('importData.loadProject') }}</template>
      <FilePicker accept=".eemproj,.zip" @change="resumeProject($event[0] ?? null)" />
      <span style="font-size: 12px; color: #909399; margin-left: 10px">{{ t('importData.loadProjectHint') }}</span>
    </el-card>

    <el-card v-if="store.imported" shadow="never">
      <template #header>{{ t('importData.currentData') }}</template>
      <el-descriptions :column="3" border size="small">
        <el-descriptions-item :label="t('importData.sampleCount')">{{ store.state.nSample }}</el-descriptions-item>
        <el-descriptions-item :label="t('importData.exRange')">{{ t('importData.rangeValue', { min: store.state.ex.min, max: store.state.ex.max, n: store.state.nex }) }}</el-descriptions-item>
        <el-descriptions-item :label="t('importData.emRange')">{{ t('importData.rangeValue', { min: store.state.em.min, max: store.state.em.max, n: store.state.nem }) }}</el-descriptions-item>
        <el-descriptions-item :label="t('importData.unit')">{{ store.state.unit }}</el-descriptions-item>
        <el-descriptions-item :label="t('importData.absorbanceData')">{{ store.state.hasAbs ? t('importData.importedState') : t('importData.notImported') }}</el-descriptions-item>
        <el-descriptions-item :label="t('importData.blank')">{{ store.state.hasBlank ? t('importData.importedState') : t('importData.notImported') }}</el-descriptions-item>
      </el-descriptions>
      <div class="toolbar" style="margin-top: 12px">
        <el-button type="primary" @click="router.push('/preprocess')">{{ t('importData.enterPreprocess') }}</el-button>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.file-rows {
  display: flex;
  flex-direction: column;
}

.file-row {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 8px 0;
  border-bottom: 1px dashed #ebeef5;
}

.file-row:last-child {
  border-bottom: none;
}

.file-label {
  width: 150px;
  flex: none;
  font-size: 13px;
  color: #606266;
}
</style>
