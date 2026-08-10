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
  api.getCarterRuntimes.mockResolvedValue({ assistant: 'Carter 1.0', carterVersion: '1.0', cloud: { configured: true, available: true }, local: { configured: true, available: false } })
  api.ingestCarterDocuments.mockResolvedValue({ documents: [] })
  api.askCarter.mockResolvedValue({ answer: 'Grounded answer', runtime: 'cloud', toolRounds: 1, assistant: 'Carter 1.0', sources: [{ documentId: 'a', documentName: 'functional.txt', sourceRef: 'unit-a' }, { documentId: 'b', documentName: 'accessibility.txt', sourceRef: 'unit-b', page: 2 }] })
})

test('shows Carter branding, cloud readiness, local unavailable, and prevents silent fallback', async () => {
  render(<CarterAsk documents={docs} />)
  expect(await screen.findByText('Ready')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Local' }))
  expect(screen.getByText('Local unavailable')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Ask Carter' })).toBeDisabled()
})

test('submits an answer and renders citations from multiple documents', async () => {
  render(<CarterAsk documents={docs} />)
  await screen.findByText('Ready')
  fireEvent.change(screen.getByLabelText('Ask Carter'), { target: { value: 'What is covered?' } })
  fireEvent.click(screen.getByRole('button', { name: 'Ask Carter' }))
  await waitFor(() => expect(api.askCarter).toHaveBeenCalledWith('What is covered?', 'cloud', ['a', 'b']))
  expect(screen.getByText('Grounded answer')).toBeInTheDocument()
  expect(screen.getByText('functional.txt')).toBeInTheDocument()
  expect(screen.getByText(/accessibility.txt/)).toBeInTheDocument()
})

test('shows an Ask failure and clears loading state', async () => {
  api.askCarter.mockRejectedValue(new Error('Carter service unavailable'))
  render(<CarterAsk documents={docs.slice(0, 1)} />)
  await screen.findByText('Ready')
  fireEvent.change(screen.getByLabelText('Ask Carter'), { target: { value: 'Question' } })
  fireEvent.click(screen.getByRole('button', { name: 'Ask Carter' }))
  expect(screen.getByRole('button', { name: /Asking Carter/ })).toBeDisabled()
  expect(await screen.findByText('Carter service unavailable')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Ask Carter' })).not.toBeDisabled()
})
