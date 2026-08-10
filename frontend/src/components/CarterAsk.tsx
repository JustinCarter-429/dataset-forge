import { useEffect, useState } from 'react'
import { MessageCircle, Sparkles } from 'lucide-react'
import { askCarter, getCarterRuntimes, ingestCarterDocuments } from '../api/client'
import type { CarterAnswer, CarterRuntime, CarterRuntimeStatus, UploadedFile } from '../api/types'

export function CarterAsk({ documents }: { documents: UploadedFile[] }) {
  const [runtime, setRuntime] = useState<CarterRuntime>('runpod')
  const [status, setStatus] = useState<CarterRuntimeStatus | null>(null)
  const [question, setQuestion] = useState(''); const [answer, setAnswer] = useState<CarterAnswer | null>(null)
  const [error, setError] = useState(''); const [busy, setBusy] = useState(false)
  useEffect(() => { getCarterRuntimes().then(setStatus).catch(() => setStatus(null)) }, [])
  const runtimeStatus = status?.runtimes[runtime]
  const available = !!runtimeStatus?.available
  const submit = async () => {
    if (!question.trim() || !documents.length || busy) return
    const pinnedRuntime = runtime
    setBusy(true); setError(''); setAnswer(null)
    try {
      const ingested = await ingestCarterDocuments(documents.map(d => d.id))
      setAnswer(await askCarter(question.trim(), pinnedRuntime, ingested.documents.map(d => d.documentId)))
    } catch (e) { setError(e instanceof Error ? e.message : 'Carter could not answer this question.') } finally { setBusy(false) }
  }
  return <section className="panel carter-panel" aria-labelledby="carter-heading"><div className="panel-title"><span className="step-badge"><Sparkles size={14} /></span><div><h2 id="carter-heading">Ask Carter 1.0</h2><p>Document-grounded answers from your local knowledge.</p></div></div><div className="runtime-row"><label><strong>Runtime</strong><select aria-label="Carter runtime" value={runtime} disabled={busy} onChange={event => setRuntime(event.target.value as CarterRuntime)}><option value="local_lm_studio">Local / LM Studio</option><option value="runpod">RunPod</option></select></label><span role="status" aria-live="polite">{available ? `Runtime: ${runtimeStatus?.label} · Ready` : `Runtime: ${runtimeStatus?.label || runtime} · Unavailable`}</span></div><textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about your source documents..." aria-label="Ask Carter" /><div className="generation-actions"><button className="generate-button" disabled={!available || !question.trim() || !documents.length || busy} onClick={submit}>{busy ? 'Asking Carter…' : 'Ask Carter'}<MessageCircle size={16} /></button></div>{error && <p className="error-banner" role="alert">{error}</p>}{answer && <div className="carter-answer" role="status" aria-live="polite"><strong>Carter 1.0</strong><small>Runtime: {answer.runtime === 'runpod' ? 'RunPod' : 'Local / LM Studio'} · Model: Carter 1.0</small><p>{answer.answer}</p><small>Sources</small><ul>{answer.sources.map(source => <li key={source.sourceRef}>{source.documentName}{source.section ? ` · ${source.section}` : ''}{source.page ? ` · page ${source.page}` : ''}</li>)}</ul></div>}</section>
}
