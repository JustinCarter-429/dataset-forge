import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import path from 'node:path'

const fixture = (name: string) => path.join(process.cwd(), 'e2e', 'fixtures', name)

test.beforeEach(async ({ page }) => { await page.request.post('http://127.0.0.1:8001/api/carter/test-reset') })

async function upload(page: import('@playwright/test').Page, name: string) {
  const input = page.locator('input[type=file]').last()
  await input.setInputFiles(fixture(name))
  await expect(page.locator('.accepted-file').first()).toBeVisible()
}

async function twoDocs(page: import('@playwright/test').Page) {
  await upload(page, 'negative-accessibility-one.txt')
  await upload(page, 'negative-accessibility-two.txt')
}

test('initial and three-document limit states are accessible', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Create a Dataset From Documents' })).toBeVisible()
  let results = await new AxeBuilder({ page }).analyze()
  expect(results.violations.filter(v => v.impact === 'serious' || v.impact === 'critical')).toHaveLength(0)
  await upload(page, 'negative-accessibility-one.txt')
  await upload(page, 'negative-accessibility-two.txt')
  await upload(page, 'negative-accessibility-one.txt')
  await expect(page.locator('.accepted-file')).toHaveCount(3)
  await expect(page.getByText('Maximum 3 source documents reached', { exact: false })).toBeVisible()
  const files = page.locator('input[type=file]')
  await expect(files.first()).toBeDisabled()
  results = await new AxeBuilder({ page }).analyze()
  expect(results.violations.filter(v => v.impact === 'serious' || v.impact === 'critical')).toHaveLength(0)
})

test('Ask Carter success includes multiple source citations', async ({ page }) => {
  await page.goto('/')
  await twoDocs(page)
  await page.getByRole('textbox', { name: 'Ask Carter' }).fill('What do these documents say about negative testing and accessibility testing?')
  await page.getByRole('button', { name: 'Ask Carter' }).click()
  await expect(page.getByText(/The selected documents describe negative testing/)).toBeVisible()
  await expect(page.locator('.carter-answer li')).toHaveCount(2)
})

test('Ask no-result, failure, and RunPod-only PoC runtime remain safe', async ({ page }) => {
  await page.goto('/')
  await twoDocs(page)
  await page.getByRole('textbox', { name: 'Ask Carter' }).fill('unsupported astronomy question')
  await page.getByRole('button', { name: 'Ask Carter' }).click()
  await expect(page.locator('.carter-answer')).toContainText('I could not find relevant information', { timeout: 15_000 })
  await page.getByRole('textbox', { name: 'Ask Carter' }).fill('TEST_ASK_FAILURE negative testing')
  await page.getByRole('button', { name: 'Ask Carter' }).click()
  await expect(page.getByRole('alert')).toContainText('could not complete')
  const runtime = page.getByLabel('Carter runtime')
  await expect(runtime).toHaveValue('runpod')
  expect(await runtime.locator('option[value="local_lm_studio"]').getAttribute('disabled')).not.toBeNull()
  await expect(page.getByRole('status')).toContainText('RunPod · Connected')
})

test('generation failure, validation failure, warning, and completed states use real API contracts', async ({ page }) => {
  await page.goto('/')
  await twoDocs(page)
  const runtimes: string[] = []
  page.on('request', request => {
    if (request.method() === 'POST' && request.url().endsWith('/api/generations')) {
      runtimes.push(request.postDataJSON().runtime)
    }
  })
  const prompt = page.locator('#prompt')
  await prompt.fill('TEST_GENERATION_FAILURE Create exactly 2 records.')
  await page.getByRole('button', { name: 'Generate Dataset' }).click()
  await expect(page.getByText('Failed', { exact: true })).toBeVisible({ timeout: 15_000 })
  expect(runtimes[0]).toBe('runpod')
  await expect(page.getByRole('button', { name: 'Download ZIP' })).toBeDisabled()
  // The failed worker releases the in-memory single-generation lease in its
  // terminal cleanup.  Wait for that backend-owned cleanup before starting
  // the next independent scenario.
  await expect.poll(async () => (await page.request.get('http://127.0.0.1:8001/api/carter/runtimes')).ok()).toBeTruthy()
  await page.waitForTimeout(150)
  await page.reload()
  await twoDocs(page)
  await prompt.fill('TEST_VALIDATION_FAILURE Create exactly 2 records.')
  await page.getByRole('button', { name: 'Generate Dataset' }).click()
  await expect(page.getByText('Failed', { exact: true })).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(150)
  await page.reload()
  await twoDocs(page)
  await prompt.fill('TEST_QUALITY_WARNING Create exactly 2 records.')
  await page.getByRole('button', { name: 'Generate Dataset' }).click()
  await expect(page.getByText('Ready', { exact: true })).toBeVisible({ timeout: 15_000 })
  // Carter review warnings remain advisory; completion is owned by the
  // validated dynamic dataset gate rather than legacy fixed-record metrics.
  await expect(page.getByText('Your package is ready')).toBeVisible()
  await expect(page.getByText('Quality check passed')).toBeVisible()
  await expect(page.getByText(/accepted .* quarantined .* rejected/)).toBeVisible()
  const downloadLink = page.getByRole('link', { name: 'Download ZIP' })
  await expect(downloadLink).toBeVisible()
  const download = page.waitForEvent('download')
  await downloadLink.click()
  expect((await download).suggestedFilename()).toMatch(/\.zip$/)
  const results = await new AxeBuilder({ page }).analyze()
  expect(results.violations.filter(v => v.impact === 'serious' || v.impact === 'critical')).toHaveLength(0)
})

test('multi-batch Carter generation exposes backend-owned batch progress', async ({ page }) => {
  await page.goto('/')
  await twoDocs(page)
  await page.locator('#prompt').fill('TEST_BATCH_PAUSE Create exactly 12 records.')
  await page.getByRole('button', { name: 'Generate Dataset' }).click()
  await expect(page.getByText('Dataset planned')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByText('Batch 2 of 3 · 5 / 12 records generated')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByLabel('5 of 12 records generated')).toBeVisible()
  await expect(page.getByText('Batch 3 of 3 · 10 / 12 records generated')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByText('12 / 12 records generated')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByText('Ready', { exact: true })).toBeVisible({ timeout: 15_000 })
  await expect(page.getByRole('link', { name: 'Download ZIP' })).toBeVisible()
})

test('keyboard reaches core controls and responsive layouts do not overflow', async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport); await page.goto('/');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy()
  }
  await page.keyboard.press('Tab')
  await expect(page.locator(':focus')).toBeVisible()
})
