import { useState, useEffect } from 'react'
import { api } from '@/hooks/useApi'
import { usePersistedState } from '@/hooks/usePersistedState'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { StatusMessage } from '@/components/status-message'
import { FormField } from '@/components/form-field'

interface ConfigFile { name: string; path: string }

interface JudgeRule {
  name: string
  description: string
  scope: 'universal' | 'sft' | 'pretrain' | string
  mode?: 'binary' | 'score'
  threshold?: number
  max_score?: number
}

interface PromptTemplate {
  system: string
  rules_header: string
  input_header_text: string
  input_header_sft_instruction: string
  input_header_sft_response: string
  trailer: string
}

interface LLMState {
  backend: string
  api_url: string
  api_key_preview: string
  api_key_set: boolean
  model: string
  samples: number
  rules: JudgeRule[]
  default_rules: JudgeRule[]
  prompt_template: PromptTemplate
  default_prompt_template: PromptTemplate
}

export default function Config() {
  // LLM API settings
  const [llm, setLlm] = useState<LLMState | null>(null)
  const [llmKey, setLlmKey] = useState('')
  const [llmDirty, setLlmDirty] = useState(false)
  const [llmStatus, setLlmStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [llmError, setLlmError] = useState<string | null>(null)

  // YAML editor
  const [files, setFiles] = useState<ConfigFile[]>([])
  const [selected, setSelected] = usePersistedState('config.selected', '')
  const [text, setText] = useState('')
  const [yamlDirty, setYamlDirty] = useState(false)
  const [yamlStatus, setYamlStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [yamlError, setYamlError] = useState<string | null>(null)

  useEffect(() => {
    api<LLMState>('/api/llm-config').then(setLlm).catch(e => setLlmError(String(e)))
    api<ConfigFile[]>('/api/configs/list').then(list => {
      setFiles(list)
      if (list.length) setSelected(list[0].path)
    }).catch(e => setYamlError(String(e)))
  }, [])

  useEffect(() => {
    if (!selected) return
    setYamlStatus('idle'); setYamlError(null)
    api<{ path: string; text: string }>(`/api/config/raw?path=${encodeURIComponent(selected)}`)
      .then(res => { setText(res.text); setYamlDirty(false) })
      .catch(e => setYamlError(e instanceof Error ? e.message : String(e)))
  }, [selected])

  const saveLLM = async () => {
    if (!llm) return
    setLlmStatus('saving'); setLlmError(null)
    try {
      const body: Record<string, unknown> = {
        backend: llm.backend,
        api_url: llm.api_url,
        model: llm.model,
        samples: llm.samples,
        rules: llm.rules,
        prompt_template: llm.prompt_template,
      }
      if (llmKey) body.api_key = llmKey
      await api('/api/llm-config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const fresh = await api<LLMState>('/api/llm-config')
      setLlm(fresh); setLlmKey(''); setLlmDirty(false); setLlmStatus('saved')
    } catch (e) {
      setLlmStatus('error')
      setLlmError(e instanceof Error ? e.message : String(e))
    }
  }

  const saveYaml = async () => {
    setYamlStatus('saving'); setYamlError(null)
    try {
      await api('/api/config/raw', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: selected, text }),
      })
      setYamlStatus('saved'); setYamlDirty(false)
    } catch (e) {
      setYamlStatus('error')
      setYamlError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Config</h2>

      {/* LLM API settings */}
      <Card>
        <CardHeader>
          <CardTitle>LLM API</CardTitle>
          <CardDescription>
            Used by the Layer-2 quality judge (pipeline and benchmark). Saved to <code>configs/llm.yaml</code>.
            Changes take effect on the next run — no server restart required.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {llm && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Backend</label>
                  <Select value={llm.backend} onValueChange={(v) => { setLlm({ ...llm, backend: v }); setLlmDirty(true) }}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="anthropic">anthropic</SelectItem>
                      <SelectItem value="openai">openai (compatible)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <FormField label="Model">
                  <Input className="font-mono text-xs" value={llm.model ?? ''}
                    onChange={e => { setLlm({ ...llm, model: e.target.value }); setLlmDirty(true) }}
                    placeholder={llm.backend === 'openai' ? 'gpt-4o-mini' : 'claude-sonnet-4-20250514'} />
                </FormField>
              </div>

              <FormField label="API URL">
                <Input className="font-mono text-xs" value={llm.api_url ?? ''}
                  onChange={e => { setLlm({ ...llm, api_url: e.target.value }); setLlmDirty(true) }}
                  placeholder="https://api.anthropic.com" />
              </FormField>

              <FormField label="API Key">
                <div className="flex items-center gap-2">
                  <Input type="password" className="font-mono text-xs" value={llmKey}
                    onChange={e => { setLlmKey(e.target.value); setLlmDirty(true) }}
                    placeholder={llm.api_key_set ? `current: ${llm.api_key_preview} (leave blank to keep)` : 'sk-...'} />
                  {llm.api_key_set && <Label className="text-xs text-green-600 whitespace-nowrap">✓ saved</Label>}
                </div>
              </FormField>

              <FormField label="Default sample size for benchmark">
                <Input type="number" value={llm.samples}
                  onChange={e => { setLlm({ ...llm, samples: Number(e.target.value) }); setLlmDirty(true) }} />
              </FormField>

              <div className="flex items-center gap-3 pt-2">
                <Button onClick={saveLLM} disabled={(!llmDirty && !llmKey) || llmStatus === 'saving'}>
                  {llmStatus === 'saving' ? 'Saving…' : 'Save LLM settings'}
                </Button>
                {llmStatus === 'saved' && !llmDirty && <span className="text-xs text-green-600">Saved</span>}
                {llmDirty && <span className="text-xs text-muted-foreground">(unsaved)</span>}
                <StatusMessage status={llmStatus === 'error' ? 'error' : 'idle'} message={llmError} />
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Prompt template */}
      <Card>
        <CardHeader>
          <CardTitle>Prompt Template</CardTitle>
          <CardDescription>
            Editable wrapping around the rules. The JSON output schema is added automatically per rule (locked) so responses stay parseable.
            Edits take effect on the next pipeline / benchmark run. Saved with LLM settings.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {llm && llm.prompt_template && (
            <>
              {(['system', 'rules_header', 'input_header_text', 'input_header_sft_instruction', 'input_header_sft_response', 'trailer'] as (keyof typeof llm.prompt_template)[]).map(key => (
                <div key={key} className="space-y-1">
                  <Label className="text-xs font-mono">{key}</Label>
                  <textarea
                    value={llm.prompt_template[key] ?? ''}
                    onChange={e => { setLlm({ ...llm, prompt_template: { ...llm.prompt_template, [key]: e.target.value } }); setLlmDirty(true) }}
                    spellCheck={false}
                    className="w-full text-xs border rounded p-2 bg-background font-mono"
                    style={{ minHeight: key === 'system' ? 70 : 36, resize: 'vertical' }}
                  />
                </div>
              ))}
              <Button variant="outline" size="sm"
                onClick={() => { setLlm({ ...llm, prompt_template: { ...llm.default_prompt_template } }); setLlmDirty(true) }}>
                Reset template to defaults
              </Button>
            </>
          )}
        </CardContent>
      </Card>

      {/* Judge rules */}
      <Card>
        <CardHeader>
          <CardTitle>Judge Rules</CardTitle>
          <CardDescription>
            Rules the LLM judge evaluates against. Edit a rule's description to tweak its prompt, delete rules you don't want applied, or add new ones.
            Scope controls where the rule applies (<code>universal</code> = both, <code>sft</code> = SFT only, <code>pretrain</code> = text-only).
            Saved alongside API settings in <code>configs/llm.yaml</code>.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {llm && (
            <>
              {llm.rules.map((r, i) => (
                <div key={i} className="rounded border p-3 space-y-2">
                  <div className="grid grid-cols-12 gap-2 items-center">
                    <Input className="col-span-3 font-mono text-xs" value={r.name}
                      onChange={e => { const rs = [...llm.rules]; rs[i] = { ...rs[i], name: e.target.value }; setLlm({ ...llm, rules: rs }); setLlmDirty(true) }}
                      placeholder="rule_name" />
                    <Select value={r.scope} onValueChange={v => { const rs = [...llm.rules]; rs[i] = { ...rs[i], scope: v }; setLlm({ ...llm, rules: rs }); setLlmDirty(true) }}>
                      <SelectTrigger className="col-span-2"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="universal">universal</SelectItem>
                        <SelectItem value="sft">sft</SelectItem>
                        <SelectItem value="pretrain">pretrain</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={r.mode ?? 'binary'} onValueChange={v => {
                      const rs = [...llm.rules]
                      const next: JudgeRule = { ...rs[i], mode: v as 'binary' | 'score' }
                      const prevDescLooksLikeDefault = !next.description ||
                        /PASS if|FAIL if|Score \d/.test(next.description)
                      if (v === 'score') {
                        next.max_score = next.max_score ?? 5
                        next.threshold = next.threshold ?? 3
                        if (prevDescLooksLikeDefault) {
                          next.description = 'Score 1..5 where 1 = clearly fails, 5 = clearly excels. Describe the criterion and what each end of the scale looks like.'
                        }
                      } else {
                        next.max_score = 1
                        next.threshold = 1
                        if (prevDescLooksLikeDefault) {
                          next.description = 'Describe the check. PASS if it holds, FAIL otherwise.'
                        }
                      }
                      rs[i] = next; setLlm({ ...llm, rules: rs }); setLlmDirty(true)
                    }}>
                      <SelectTrigger className="col-span-2"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="binary">binary</SelectItem>
                        <SelectItem value="score">score</SelectItem>
                      </SelectContent>
                    </Select>
                    {r.mode === 'score' ? (
                      <div className="col-span-4 flex items-center gap-1 text-xs">
                        <span className="text-muted-foreground">pass ≥</span>
                        <Input type="number" step="0.5" className="h-8 w-16" value={r.threshold ?? 3}
                          onChange={e => { const rs = [...llm.rules]; rs[i] = { ...rs[i], threshold: Number(e.target.value) }; setLlm({ ...llm, rules: rs }); setLlmDirty(true) }} />
                        <span className="text-muted-foreground">/</span>
                        <Input type="number" className="h-8 w-16" value={r.max_score ?? 5}
                          onChange={e => { const rs = [...llm.rules]; rs[i] = { ...rs[i], max_score: Number(e.target.value) }; setLlm({ ...llm, rules: rs }); setLlmDirty(true) }} />
                      </div>
                    ) : <div className="col-span-4" />}
                    <Button variant="ghost" size="sm" className="col-span-1 text-destructive"
                      onClick={() => { const rs = llm.rules.filter((_, idx) => idx !== i); setLlm({ ...llm, rules: rs }); setLlmDirty(true) }}>
                      Delete
                    </Button>
                  </div>
                  <textarea
                    value={r.description}
                    onChange={e => { const rs = [...llm!.rules]; rs[i] = { ...rs[i], description: e.target.value }; setLlm({ ...llm!, rules: rs }); setLlmDirty(true) }}
                    spellCheck={false}
                    className="w-full text-xs border rounded p-2 bg-background"
                    style={{ minHeight: 60 }}
                    placeholder="Rule description shown to the judge (e.g. 'Is the text well-structured? PASS if coherent, FAIL if fragmented.')"
                  />
                </div>
              ))}
              <div className="flex gap-2">
                <Button variant="outline" size="sm"
                  onClick={() => { setLlm({ ...llm!, rules: [...llm!.rules, { name: 'new_rule', description: '', scope: 'universal' }] }); setLlmDirty(true) }}>
                  + Add rule
                </Button>
                <Button variant="outline" size="sm"
                  onClick={() => { setLlm({ ...llm!, rules: llm!.default_rules }); setLlmDirty(true) }}>
                  Reset to defaults
                </Button>
                <span className="text-xs text-muted-foreground self-center">
                  Changes saved when you click <b>Save LLM settings</b> above.
                </span>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* YAML editor */}
      <Card>
        <CardHeader>
          <CardTitle>YAML Editor</CardTitle>
          <CardDescription>Edit pipeline configs under <code>configs/</code>. Saved files take effect on the next run.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-3">
            <Select value={selected} onValueChange={setSelected}>
              <SelectTrigger className="w-[280px]"><SelectValue placeholder="Pick a config…" /></SelectTrigger>
              <SelectContent>
                {files.map(f => <SelectItem key={f.path} value={f.path}>{f.name}</SelectItem>)}
              </SelectContent>
            </Select>
            <Button onClick={saveYaml} disabled={!yamlDirty || yamlStatus === 'saving'} size="sm">
              {yamlStatus === 'saving' ? 'Saving…' : 'Save'}
            </Button>
            {yamlStatus === 'saved' && !yamlDirty && <span className="text-xs text-green-600">Saved</span>}
            {yamlDirty && <span className="text-xs text-muted-foreground">(unsaved)</span>}
          </div>

          <textarea
            value={text}
            onChange={e => { setText(e.target.value); setYamlDirty(true); setYamlStatus('idle') }}
            spellCheck={false}
            className="w-full font-mono text-xs border rounded p-3 bg-muted/30"
            style={{ minHeight: '50vh' }}
          />

          <StatusMessage status={yamlStatus === 'error' ? 'error' : 'idle'} message={yamlError} />
        </CardContent>
      </Card>
    </div>
  )
}
