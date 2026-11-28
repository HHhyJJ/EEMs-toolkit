<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps<{ job: any }>()

const percent = computed(() => {
  if (!props.job?.total) return 0
  return Math.round((props.job.done / props.job.total) * 100)
})

const statusText = computed(() => {
  const state = props.job?.state
  return {
    running: t('job.running'),
    done: t('job.done'),
    stopped: t('job.stopped'),
    error: t('job.error'),
    stale: t('job.stale')
  }[state] ?? state
})
</script>

<template>
  <el-card v-if="job" class="section-card" shadow="never">
    <template #header>
      <div style="display: flex; align-items: center; justify-content: space-between">
        <span>
          <span class="status-dot" :class="job.state"></span>
          {{ job.kind }} · {{ statusText }}
        </span>
        <span style="font-size: 12px; color: #909399">
          {{ job.done }} / {{ job.total }} · {{ job.elapsed }} s
        </span>
      </div>
    </template>
    <el-progress :percentage="percent" :status="job.state === 'error' ? 'exception' : job.state === 'done' ? 'success' : undefined" />
    <div v-if="job.error" style="color: #f56c6c; margin-top: 8px; font-size: 13px">{{ job.error }}</div>
    <div class="mono" style="margin-top: 10px">{{ (job.log_tail || []).join('\n') || t('job.waitingLogs') }}</div>
  </el-card>
</template>
