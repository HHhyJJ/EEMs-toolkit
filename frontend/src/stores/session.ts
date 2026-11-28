import { defineStore } from 'pinia'
import api from '../api'
import { i18n } from '../i18n'

interface HistoryState {
  position: number
  entries: any[]
  max: number
  canRedo: boolean
}

export const useSessionStore = defineStore('session', {
  state: () => ({
    state: {} as any,
    history: { position: -1, entries: [], max: 50, canRedo: false } as HistoryState,
    samples: [] as any[],
    job: null as any,
    jobId: '',
    eventSource: null as EventSource | null
  }),
  getters: {
    imported: (state) => Boolean(state.state?.imported),
    running: (state) => state.job?.state === 'running',
    sampleFiles: (state) => state.samples.map((s) => s.file)
  },
  actions: {
    async refreshState() {
      this.state = (await api.get('/api/state')).data
    },
    async refreshHistory() {
      this.history = (await api.get('/api/history')).data
    },
    async refreshSamples() {
      if (!this.state?.imported) {
        this.samples = []
        return
      }
      this.samples = (await api.get('/api/samples')).data.samples
    },
    async bootstrap() {
      await api.get('/api/session')
      await this.refreshState()
      await this.refreshHistory()
      await this.refreshSamples()
      await this.refreshActiveJob()
    },
    async refreshActiveJob() {
      const { data } = await api.get('/api/jobs')
      const active = data?.active
      if (!active) {
        this.job = null
        this.jobId = ''
        this.closeJobStream()
        return
      }
      this.job = active
      this.jobId = active.id
      if (active.state === 'running') this.listenJob(active.id)
    },
    applyResult(data: any) {
      if (data?.state) this.state = data.state
      if (data?.history) this.history = data.history
    },
    async apply(url: string, payload?: any) {
      const response = await api.post(url, payload ?? {})
      this.applyResult(response.data)
      await this.refreshSamples()
      return response.data
    },
    async jump(index: number) {
      const response = await api.post('/api/history/jump', { index })
      this.applyResult(response.data)
      await this.refreshSamples()
    },
    async trim(index: number) {
      const response = await api.post('/api/history/trim', { index })
      this.applyResult(response.data)
      await this.refreshSamples()
    },
    async startJob(kind: 'parafac' | 'split', payload: any) {
      const { data } = await api.post(`/api/jobs/${kind}`, payload)
      this.jobId = data.job_id
      this.job = { id: data.job_id, kind, state: 'running', done: 0, total: data.total, log_tail: [] }
      this.listenJob(data.job_id)
      return data.job_id
    },
    listenJob(jobId: string) {
      this.closeJobStream()
      const source = new EventSource(`/api/jobs/${jobId}/events`)
      this.eventSource = source
      source.onmessage = async (event) => {
        const job = JSON.parse(event.data)
        this.job = job
        if (job.state !== 'running') {
          source.close()
          this.eventSource = null
          await this.refreshState()
        }
      }
      source.onerror = async () => {
        source.close()
        this.eventSource = null
        try {
          const { data } = await api.get(`/api/jobs/${jobId}`)
          this.job = data
          if (data?.state === 'running') {
            setTimeout(() => {
              if (this.jobId === jobId && this.job?.state === 'running') this.listenJob(jobId)
            }, 1500)
          } else {
            await this.refreshState()
          }
        } catch {
          if (this.job) this.job = { ...this.job, state: 'error', error: i18n.global.t('common.connectionLost') }
        }
      }
    },
    async stopJob() {
      if (this.jobId) await api.post(`/api/jobs/${this.jobId}/stop`)
    },
    closeJobStream() {
      this.eventSource?.close()
      this.eventSource = null
    }
  }
})
