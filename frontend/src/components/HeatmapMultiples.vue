<script setup lang="ts">
import EemHeatmap from './EemHeatmap.vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

defineProps<{
  items: { title: string; ex: number[]; em: number[]; z: (number | null)[][]; zmin?: number; zmax?: number }[]
  height?: string
}>()
</script>

<template>
  <div class="heatmap-grid">
    <EemHeatmap
      v-for="item in items"
      :key="item.title"
      :title="item.title"
      :ex="item.ex"
      :em="item.em"
      :z="item.z"
      :zmin="item.zmin"
      :zmax="item.zmax"
      :height="height"
    />
  </div>
</template>
