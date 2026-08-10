import { useEffect, useState } from 'react'
import { MessageCircle, Sparkles } from 'lucide-react'
import { askCarter, getCarterRuntimes, ingestCarterDocuments } from '../api/client'
import type { CarterAnswer, CarterRuntimeStatus, UploadedFile } from '../api/types'

export function CarterAsk({ documents }: { documents: UploadedFile[] }) {
  const [runtime, setRuntime] = useState<'cloud' | 'local'>('cloud')
  const [status, setStatus] = useState<CarterRuntimeStatus | null>(null)
  const [question, setQuestion] = useState(''); const [answer, setAnswer] = useState<CarterAnswer | null>(null)
  const [error, setError] = useState(''); const [busy, setBusy] = useState(false)
  useEffect(() => { getCarterRuntimes().then(setStatus).catch(() => setStatus(null)) }, [])
  const available = runtime === 'local' ? !!status?.local.available : !!status?.cloud.available
  const submit = async () => { if (!question.trim() || !documents.length || busy) return; setBusy(true); setError(''); setAnswer(null); try { await ingestCarterDocuments(documents.map(d => d.id)); setAnswer(await askCarter(question.trim(), runtime, documents.map(d => d.id))) } catch (e) { setError(e instanceof Error ? e.message : 'Carter could not answer this question.') } finally { setBusy(false) } }
  return <section className="panel carter-panel"><div className="panel-title"><span className="step-badge"><Sparkles size={14} /></span><div><h2>Ask Carter 1.0</h2><p>Document-grounded answers from your local knowledge.</p></div></div><div className="runtime-row"><strong>Carter 1.0 runtime</strong><div role="group" aria-label="Carter runtime"><button className={runtime === 'cloud' ? 'runtime-selected' : ''} onClick={() => setRuntime('cloud')}>Cloud</button><button className={runtime === 'local' ? 'runtime-selected' : ''} onClick={() => setRuntime('local')}>Local</button></div><span>{available ? 'Ready' : runtime === 'local' ? 'Local unavailable' : 'Cloud unavailable'}</span></div><textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about your source documents..." aria-label="Ask Carter" /><div className="generation-actions"><button className="generate-button" disabled={!available || !question.trim() || !documents.length || busy} onClick={submit}>{busy ? 'Asking Carter…' : 'Ask Carter'}<MessageCircle size={16} /></button></div>{error && <p className="error-banner">{error}</p>}{answer && <div className="carter-answer"><strong>Carter 1.0</strong><p>{answer.answer}</p><small>Sources</small><ul>{answer.sources.map(source => <li key={source.sourceRef}>{source.documentName}{source.section ? ` · ${source.section}` : ''}{source.page ? ` · page ${source.page}` : ''}</li>)}</ul></div>}</section>
}
