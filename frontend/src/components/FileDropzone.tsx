import { useRef, useState } from 'react'
import { FileText, UploadCloud, X } from 'lucide-react'

const allowed = ['.pdf', '.docx', '.txt']
const readable = (bytes: number) => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`

export function FileDropzone({ file, onChange, error, busy }: { file: File | null, onChange: (file: File | null, error?: string) => void, error?: string, busy?: boolean }) {
  const input = useRef<HTMLInputElement>(null); const [active, setActive] = useState(false)
  const pick = (candidate?: File) => { if (!candidate) return; const ext = candidate.name.slice(candidate.name.lastIndexOf('.')).toLowerCase(); if (!allowed.includes(ext)) return onChange(null, 'That file type is not supported. Choose a PDF, DOCX, or TXT file.'); if (candidate.size === 0) return onChange(null, 'That file is empty. Choose a document with content.'); if (candidate.size > 25 * 1024 * 1024) return onChange(null, 'That file is larger than the 25 MB limit.'); onChange(candidate) }
  return <div><div className={`dropzone ${active ? 'dropzone-active' : ''} ${error ? 'dropzone-error' : ''}`} role="button" tabIndex={0} aria-label="Upload PDF, DOCX, or TXT document" onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.current?.click() } }} onDragOver={e => { e.preventDefault(); setActive(true) }} onDragLeave={() => setActive(false)} onDrop={e => { e.preventDefault(); setActive(false); pick(e.dataTransfer.files[0]) }}>
    {file ? <div className="file-selected"><div className="file-icon"><FileText size={24} /></div><div className="file-details"><strong>{file.name}</strong><span>{readable(file.size)} · {file.name.split('.').pop()?.toUpperCase()} document · {busy ? 'Uploading…' : 'Ready for Phase 1 pipeline'}</span></div><button className="icon-button" onClick={() => onChange(null)} aria-label="Remove file"><X size={18} /></button></div> : <><div className="upload-icon"><UploadCloud size={25} /></div><strong>Drop your document here</strong><span>PDF, DOCX, or TXT · one file up to 25 MB</span><button className="browse-button" onClick={() => input.current?.click()}>Browse Files</button></>}
    <input ref={input} type="file" accept=".pdf,.docx,.txt" hidden onChange={e => pick(e.target.files?.[0])} />
  </div>{error && <p className="field-error">{error}</p>}</div>
}
