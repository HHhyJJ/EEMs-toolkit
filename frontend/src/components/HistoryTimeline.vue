<script setup lang="ts">
import { ElMessageBox } from 'element-plus'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

defineProps<{
  entries: any[]
  position: number
}>()

const emit = defineEmits<{
  (event: 'jump', index: number): void
  (event: 'trim', index: number): void
}>()

const labels = computed<Record<string, string>>(() => ({
  import: t('history.importData'),
  inner_effect_correct: t('history.innerEffectCorrect'),
  minus_blank: t('history.minusBlank'),
  raman_areal: t('history.ramanAreal'),
  sub_dataset: t('history.subDataset'),
  dilute: t('history.dilute'),
  smooth: t('history.smooth'),
  cut_scatter: t('history.cutScatter'),
  interpolate: t('history.interpolate'),
  exclude_samples: t('history.excludeSamples')
}))

async function remove(index: number) {
  try {
    await ElMessageBox.confirm(
      t('history.trimConfirm', { index, previous: index - 1 }),
      t('history.trimTitle'),
      { type: 'warning', confirmButtonText: t('history.delete'), cancelButtonText: t('history.cancel') }
    )
    emit('trim', index)
  } catch {
    // user cancelled
  }
}
</script>

<template>
  <div class="history-wrap">
    <el-timeline>
      <el-timeline-item
        v-for="entry in entries"
        :key="entry.index"
        :timestamp="entry.timestamp"
        placement="top"
        :type="entry.index === position ? 'primary' : entry.index > position ? 'info' : 'success'"
      >
        <div
          class="history-item"
          :class="{ active: entry.index === position, future: entry.index > position }"
          @click="emit('jump', entry.index)"
        >
          <span>#{{ entry.index }} {{ labels[entry.action] ?? entry.action }}</span>
          <el-button
            v-if="entry.index > 0"
            link
            type="danger"
            size="small"
            @click.stop="remove(entry.index)"
          >
            {{ t('history.deleteFollowing') }}
          </el-button>
        </div>
      </el-timeline-item>
    </el-timeline>
  </div>
</template>

<style scoped>
.history-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  cursor: pointer;
}

.history-item.future {
  opacity: 0.55;
}
</style>
