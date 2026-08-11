import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import { CarterAsk } from './CarterAsk'

const api = vi.hoisted(() => ({
  askCarter: vi.fn(), getCarterRuntimes: vi.fn(), ingestCarterDocuments: vi.fn(),
}))
vi.mock('../api/client', () => api)

const docs = [
  { id: 'a', name: 'functional.txt', sizeBytes: 12, mimeType: 'text/plain', extension: 'txt', status: 'ready' as const },
  { id: 'b', name: 'accessibility.txt', sizeBytes: 12, mimeType: 'text/plain', extension: 'txt', status: 'ready' as const },
]

beforeEach(() => {
  api.getCarterRuntimes.mockResolvedValue({ assistant: 'Carter 1.0', carterVersion: '1.0', runtimes: { runpod: { configured: true, available: true, label: 'RunPod' }, local_lm_studio: { configured: true, available: false, label: 'Local / LM Studio' } } })
  api.ingestCarterDocuments.mockResolvedValue({ documents: [{ documentId: 'a', name: 'functional.txt' }, { documentId: 'b', name: 'accessibility.txt' }] })
  api.askCarter.mockResolvedValue({ answer: 'Grounded answer', runtime: 'runpod', logicalModel: 'Carter 1.0', technicalModel: 'openai/gpt-oss-20b', inferenceCount: 1, toolRounds: 1, assistant: 'Carter 1.0', sources: [{ documentId: 'a', documentName: 'functional.txt', sourceRef: 'unit-a' }, { documentId: 'b', documentName: 'accessibility.txt', sourceRef: 'unit-b', page: 2 }] })
})

test('locks the PoC runtime to RunPod while keeping LM Studio visible but disabled', async () => {
  render(<CarterAsk documents={docs} />)
  expect(await screen.findByRole('status')).toHaveTextContent('RunPod')
  const selector = screen.getByLabelText('Carter runtime') as HTMLSelectElement
  expect(selector.value).toBe('runpod')
  expect(screen.getByRole('option', { name: /Local \/ LM Studio/ })).toBeDisabled()
  fireEvent.change(screen.getByLabelText('Ask Carter'), { target: { value: 'Question' } })
  expect(screen.getByRole('button', { name: 'Ask Carter' })).not.toBeDisabled()
})

test('submits an answer and renders citations from multiple documents', async () => {
  render(<CarterAsk documents={docs} />)
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('RunPod'))
  fireEvent.change(screen.getByLabelText('Ask Carter'), { target: { value: 'What is covered?' } })
  fireEvent.click(screen.getByRole('button', { name: 'Ask Carter' }))
  await waitFor(() => expect(api.askCarter).toHaveBeenCalledWith('What is covered?', 'runpod', ['a', 'b']))
  expect(screen.getByText('Grounded answer')).toBeInTheDocument()
  expect(screen.getByText('functional.txt')).toBeInTheDocument()
  expect(screen.getByText(/accessibility.txt/)).toBeInTheDocument()
})

test('shows an Ask failure and clears loading state', async () => {
  api.askCarter.mockRejectedValue(new Error('Carter service unavailable'))
  render(<CarterAsk documents={docs.slice(0, 1)} />)
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('RunPod'))
  fireEvent.change(screen.getByLabelText('Ask Carter'), { target: { value: 'Question' } })
  fireEvent.click(screen.getByRole('button', { name: 'Ask Carter' }))
  expect(screen.getByRole('button', { name: /Asking Carter/ })).toBeDisabled()
  expect(await screen.findByText('Carter service unavailable')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Ask Carter' })).not.toBeDisabled()
})
