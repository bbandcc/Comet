import client from './client'

interface Wrapped<T> {
  code: number
  message: string
  data: T
}

export type TriggerType = 'daily' | 'weekly' | 'interval'

export interface AgentTask {
  id: string
  name: string
  instruction: string
  kb_ids: string[]
  trigger_type: TriggerType
  trigger_time: string | null
  trigger_weekday: number | null
  trigger_interval_hours: number | null
  enabled: boolean
  notify_enabled: boolean
  last_run_at: string | null
  last_status: 'running' | 'done' | 'failed' | null
  next_run_at: string | null
  created_at: string | null
}

export interface AgentTaskUpsert {
  name: string
  instruction: string
  kb_ids?: string[] | null
  trigger_type: TriggerType
  trigger_time?: string | null
  trigger_weekday?: number | null
  trigger_interval_hours?: number | null
  enabled: boolean
  notify_enabled?: boolean
}

export interface AgentTaskRun {
  id: string
  title: string
  status: 'pending' | 'planning' | 'searching' | 'writing' | 'summarizing' | 'done' | 'failed'
  error_msg: string | null
  created_at: string | null
  // verified 保持旧 LoopRun.status 值域；quality_status 是 S3a 起的质量真值。
  verified?: 'running' | 'passed' | 'failed' | 'exceeded' | 'none'
  quality_status?: 'passed' | 'failed_quality' | 'judge_error' | 'unavailable' | 'skipped' | 'none'
  final_score?: number | null
}

export const agentTaskApi = {
  list() {
    return client.get<unknown, Wrapped<AgentTask[]>>('/agent-tasks')
  },
  create(body: AgentTaskUpsert) {
    return client.post<unknown, Wrapped<AgentTask>>('/agent-tasks', body)
  },
  update(id: string, body: AgentTaskUpsert) {
    return client.put<unknown, Wrapped<AgentTask>>(`/agent-tasks/${id}`, body)
  },
  setEnabled(id: string, enabled: boolean) {
    return client.patch<unknown, Wrapped<AgentTask>>(`/agent-tasks/${id}/enabled`, {
      enabled,
    })
  },
  runNow(id: string) {
    return client.post<unknown, Wrapped<null>>(`/agent-tasks/${id}/run`)
  },
  remove(id: string) {
    return client.delete<unknown, Wrapped<null>>(`/agent-tasks/${id}`)
  },
  runs(id: string) {
    return client.get<unknown, Wrapped<AgentTaskRun[]>>(`/agent-tasks/${id}/runs`)
  },
  unreadCount() {
    return client.get<unknown, Wrapped<{ count: number }>>('/agent-tasks/unread-count')
  },
  markSeen() {
    return client.post<unknown, Wrapped<null>>('/agent-tasks/mark-seen')
  },
}
