import { createRouter, createWebHashHistory } from 'vue-router'

import ImportView from './views/ImportView.vue'
import PreprocessView from './views/PreprocessView.vue'
import ParafacModelView from './views/ParafacModelView.vue'
import OutlierView from './views/OutlierView.vue'
import SplitView from './views/SplitView.vue'
import ParafacResultView from './views/ParafacResultView.vue'
import IndicesView from './views/IndicesView.vue'
import ExportView from './views/ExportView.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/import' },
    { path: '/import', component: ImportView, meta: { titleKey: 'menu.import', step: 1 } },
    { path: '/preprocess', component: PreprocessView, meta: { titleKey: 'menu.preprocess', step: 2 } },
    { path: '/parafac', redirect: '/parafac/model' },
    { path: '/parafac/model', component: ParafacModelView, meta: { titleKey: 'menu.model', step: 3 } },
    { path: '/parafac/outlier', component: OutlierView, meta: { titleKey: 'menu.outlier', step: 3 } },
    { path: '/parafac/validation', component: SplitView, meta: { titleKey: 'menu.validation', step: 3 } },
    { path: '/parafac/result', component: ParafacResultView, meta: { titleKey: 'menu.result', step: 3 } },
    { path: '/indices', component: IndicesView, meta: { titleKey: 'menu.indices', step: 4 } },
    { path: '/export', component: ExportView, meta: { titleKey: 'menu.export', step: 5 } }
  ]
})

export default router
