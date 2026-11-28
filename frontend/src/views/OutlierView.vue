<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useI18n } from 'vue-i18n'
import api, { errorMessage } from '../api'
import { useSessionStore } from '../stores/session'
import OutlierScatter from '../components/OutlierScatter.vue'

const { t } = useI18n()
const store = useSessionStore()
const router = useRouter()

const selectedF = ref<number | null>(null)
const outlier = ref<any>(null)
const selectedOrders = ref<number[]>([])

async function loadOutlier() {
  if (!selectedF.value) {
    outlier.value = null
    return
  }
  try {
    const { data } = await api.get('/api/charts/outlier', { params: { f: selectedF.value } })
    outlier.value = data
    selectedOrders.value = []
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function excludeSelected() {
  if (!selectedOrders.value.length) {
    ElMessage.warning(t('outlier.selectSamples'))
    return
  }
  try {
    await ElMessageBox.confirm(
      t('outlier.removeConfirm', { count: selectedOrders.value.length }),
      t('outlier.removeConfirmTitle'),
      { type: 'warning' }
    )
  } catch {
    return
  }
  try {
    await store.apply('/api/samples/exclude', { orders: selectedOrders.value })
    outlier.value = null
    selectedF.value = null
    ElMessage.success(t('outlier.removeSuccess'))
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

watch(selectedF, loadOutlier)
watch(() => store.state.facNums, (nums: number[]) => {
  if (!nums || !nums.length) {
    selectedF.value = null
    return
  }
  if (!selectedF.value || !nums.includes(selectedF.value)) selectedF.value = nums[0]
}, { deep: true, immediate: true })

onMounted(async () => {
  if (store.state.facNums?.length) await loadOutlier()
})
</script>

<template>
  <div>
    <h2 class="page-title">{{ t('outlier.title') }}</h2>
    <p class="page-hint">
      {{ t('outlier.hint') }}
    </p>

    <el-alert
      type="info"
      show-icon
      :closable="false"
      :title="t('outlier.recommendation')"
      style="margin-bottom: 12px"
    />

    <template v-if="store.state.facNums && store.state.facNums.length">
      <el-card shadow="never" class="section-card">
        <div class="toolbar">
          <span>{{ t('outlier.componentCount') }}</span>
          <el-select v-model="selectedF" size="small" style="width: 120px">
            <el-option v-for="n in store.state.facNums" :key="n" :label="`f = ${n}`" :value="n" />
          </el-select>
          <el-tag v-if="store.state.coreConsistency && selectedF" type="info">
            CC {{ (store.state.coreConsistency[String(selectedF)] ?? 0).toFixed(2) }} ·
            {{ t('outlier.explanationRate') }} {{ ((store.state.explanationRate[String(selectedF)] ?? 0) * 100).toFixed(2) }}%
          </el-tag>
          <el-button type="danger" size="small" :disabled="!selectedOrders.length" @click="excludeSelected">
            {{ t('outlier.excludeSelected') }}
          </el-button>
          <el-button size="small" @click="router.push('/parafac/model')">{{ t('outlier.backToModel') }}</el-button>
        </div>
      </el-card>

      <el-card v-if="outlier" shadow="never">
        <template #header>{{ t('outlier.scatterTitle') }}</template>
        <OutlierScatter :samples="outlier.samples" @change="selectedOrders = $event" />
      </el-card>
    </template>
    <el-empty v-else :description="t('outlier.empty')">
      <el-button type="primary" @click="router.push('/parafac/model')">{{ t('outlier.goToModel') }}</el-button>
    </el-empty>
  </div>
</template>
