<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

const props = defineProps<{
  multiple?: boolean
  accept?: string
}>()

const emit = defineEmits<{ (event: 'change', files: File[]): void }>()

const { t } = useI18n()
const input = ref<HTMLInputElement | null>(null)
const files = ref<File[]>([])

const label = computed(() => {
  if (!files.value.length) return t('common.noFileChosen')
  if (!props.multiple) return files.value[0].name
  return t('common.filesChosen', { n: files.value.length })
})

function open() {
  input.value?.click()
}

function onChange(event: Event) {
  const list = (event.target as HTMLInputElement).files
  files.value = list ? Array.from(list) : []
  emit('change', files.value)
}
</script>

<template>
  <div class="file-picker">
    <el-button size="small" @click="open">{{ multiple ? t('common.chooseFiles') : t('common.chooseFile') }}</el-button>
    <input
      ref="input"
      class="file-picker-input"
      type="file"
      :multiple="multiple"
      :accept="accept"
      @change="onChange"
    />
    <span class="file-picker-name" :class="{ empty: !files.length }">{{ label }}</span>
  </div>
</template>

<style scoped>
.file-picker {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.file-picker-input {
  display: none;
}

.file-picker-name {
  font-size: 12px;
  color: #606266;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-picker-name.empty {
  color: #909399;
}
</style>
