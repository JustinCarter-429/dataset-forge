import type { CarterAnswer, CarterRuntimeStatus, GenerationJob, GenerationResult, OutputFormat, UploadedFile } from './types'

const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api'

export async function generateDataset(file: File, prompt: string, format: OutputFormat): Promise<GenerationResult> {
  const body = new FormData(); body.append('file', file); body.append('dataset_prompt', prompt); body.append('output_format', format)
  const response = await fetch(`${baseUrl}/generate`, { method: 'POST', body })
  if (!response.ok) { const data = await response.json().catch(() => null); throw new Error(data?.detail || 'The backend could not create your dataset.') }
  return response.json()
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, init)
  if (!response.ok) { const data = await response.json().catch(() => null); throw new Error(data?.detail || 'The backend could not complete the request.') }
  return response.json()
}

export async function uploadFile(file: File): Promise<UploadedFile> {
  const body = new FormData(); body.append('file', file)
  const data = await request<{ file: UploadedFile }>('/files', { method: 'POST', body }); return data.file
}

export async function createGeneration(fileId: string, datasetPrompt: string, outputFormat: OutputFormat, fileIds: string[] = [fileId]) {
  return request<{ generationId: string; status: string }>('/generations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ fileId, fileIds, datasetPrompt, outputFormat }) })
}

export function getGeneration(id: string) { return request<GenerationJob>(`/generations/${encodeURIComponent(id)}`) }
export function cancelGeneration(id: string) { return request<GenerationJob>(`/generations/${encodeURIComponent(id)}/cancel`, { method: 'POST' }) }

export function downloadUrl(jobId: string) { return `${baseUrl}/download/${encodeURIComponent(jobId)}` }
export function generationDownloadUrl(id: string) { return `${baseUrl}/generations/${encodeURIComponent(id)}/download` }
export function getCarterRuntimes() { return request<CarterRuntimeStatus>('/carter/runtimes') }
export function ingestCarterDocuments(fileIds: string[]) { return request<{ documents: { documentId: string; name: string }[] }>('/carter/ingest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ fileIds }) }) }
export function askCarter(question: string, runtime: 'cloud' | 'local', documentIds: string[]) { return request<CarterAnswer>('/carter/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, runtime, documentIds }) }) }
