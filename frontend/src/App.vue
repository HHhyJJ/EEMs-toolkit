<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import en from 'element-plus/es/locale/lang/en'
import { setLocale, type AppLocale } from './i18n'
import { useSessionStore } from './stores/session'

const store = useSessionStore()
const route = useRoute()
const { t, locale } = useI18n()
const active = computed(() => route.path)
const elementLocale = computed(() => (locale.value === 'en-US' ? en : zhCn))

function changeLocale(value: AppLocale) {
  setLocale(value)
}

function updateTitle() {
  const key = route.meta.titleKey as string | undefined
  document.title = key ? `${t(key)} · EEMs-toolkit` : 'EEMs-toolkit'
}

watch(() => [route.path, locale.value], updateTitle)

onMounted(() => {
  document.documentElement.lang = locale.value
  updateTitle()
  store.bootstrap().catch(() => undefined)
})
</script>

<template>
  <el-config-provider :locale="elementLocale">
    <el-container class="app-shell">
      <el-header class="app-header" height="56px">
        <div class="brand">
          EEMs-toolkit <small>{{ t('app.subtitle') }}</small>
        </div>
        <div class="header-meta">
          <el-tag v-if="store.imported" type="success" effect="dark">
            {{ store.state.nSample }} {{ t('app.samples') }} · {{ store.state.nex }} Ex × {{ store.state.nem }} Em ·
            {{ store.state.unit }}
          </el-tag>
          <el-tag v-else type="info" effect="dark">{{ t('app.noData') }}</el-tag>
          <el-tag v-if="store.imported" type="warning" effect="dark">
            {{ t('app.history') }} {{ store.history.position + 1 }} / {{ store.history.entries.length }}
          </el-tag>
          <el-tag v-if="store.job" :type="store.job.state === 'done' ? 'success' : store.job.state === 'running' ? 'warning' : 'info'">
            {{ store.job.kind }}: {{ store.job.state }}
          </el-tag>
          <el-select
            :model-value="locale"
            size="small"
            style="width: 112px"
            :aria-label="t('app.language')"
            @update:model-value="changeLocale"
          >
            <el-option label="中文" value="zh-CN" />
            <el-option label="English" value="en-US" />
          </el-select>
        </div>
      </el-header>
      <el-container>
        <el-aside width="212px" class="app-aside">
          <el-menu :default-active="active" :default-openeds="['parafac']" router>
            <el-menu-item index="/import">{{ t('menu.import') }}</el-menu-item>
            <el-menu-item index="/preprocess" :disabled="!store.imported">{{ t('menu.preprocess') }}</el-menu-item>
            <el-sub-menu index="parafac" :disabled="!store.imported">
              <template #title>{{ t('menu.parafac') }}</template>
              <el-menu-item index="/parafac/model">{{ t('menu.model') }}</el-menu-item>
              <el-menu-item index="/parafac/outlier">{{ t('menu.outlier') }}</el-menu-item>
              <el-menu-item index="/parafac/validation">{{ t('menu.validation') }}</el-menu-item>
              <el-menu-item index="/parafac/result">{{ t('menu.result') }}</el-menu-item>
            </el-sub-menu>
            <el-menu-item index="/indices" :disabled="!store.imported">{{ t('menu.indices') }}</el-menu-item>
            <el-menu-item index="/export" :disabled="!store.imported">{{ t('menu.export') }}</el-menu-item>
          </el-menu>
        </el-aside>
        <el-main class="app-main">
          <router-view />
        </el-main>
      </el-container>
    </el-container>
  </el-config-provider>
</template>
