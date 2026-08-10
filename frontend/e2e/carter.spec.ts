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

test('Ask no-result, failure, and local isolation remain safe', async ({ page }) => {
  await page.goto('/')
  await twoDocs(page)
  await page.getByRole('textbox', { name: 'Ask Carter' }).fill('unsupported astronomy question')
  await page.getByRole('button', { name: 'Ask Carter' }).click()
  await expect(page.locator('.carter-answer')).toContainText('I could not find relevant information', { timeout: 15_000 })
  await page.getByRole('textbox', { name: 'Ask Carter' }).fill('TEST_ASK_FAILURE negative testing')
  await page.getByRole('button', { name: 'Ask Carter' }).click()
  await expect(page.getByRole('alert')).toContainText('could not complete')
  await page.getByLabel('Carter runtime').selectOption('local_lm_studio')
  await expect(page.getByRole('status')).toContainText('Local / LM Studio · Unavailable')
  await expect(page.getByRole('button', { name: 'Ask Carter' })).toBeDisabled()
})

test('generation failure, validation failure, warning, and completed states use real API contracts', async ({ page }) => {
  await page.goto('/')
  await twoDocs(page)
  const prompt = page.locator('#prompt')
  await prompt.fill('TEST_GENERATION_FAILURE Create exactly 2 records.')
  await page.getByRole('button', { name: 'Generate Dataset' }).click()
  await expect(page.getByText('Failed', { exact: true })).toBeVisible({ timeout: 15_000 })
  await expect(page.getByRole('button', { name: 'Download ZIP' })).toBeDisabled()
  await page.reload()
  await twoDocs(page)
  await prompt.fill('TEST_VALIDATION_FAILURE Create exactly 2 records.')
  await page.getByRole('button', { name: 'Generate Dataset' }).click()
  await expect(page.getByText('Failed', { exact: true })).toBeVisible({ timeout: 15_000 })
  await page.reload()
  await twoDocs(page)
  await prompt.fill('TEST_QUALITY_WARNING Create exactly 2 records.')
  await page.getByRole('button', { name: 'Generate Dataset' }).click()
  await expect(page.getByText('Ready', { exact: true })).toBeVisible({ timeout: 15_000 })
  await expect(page.getByText(/completed with warnings/)).toBeVisible()
  await expect(page.getByRole('button', { name: 'Download ZIP' })).toBeEnabled()
  const results = await new AxeBuilder({ page }).analyze()
  expect(results.violations.filter(v => v.impact === 'serious' || v.impact === 'critical')).toHaveLength(0)
})

test('keyboard reaches core controls and responsive layouts do not overflow', async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport); await page.goto('/');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy()
  }
  await page.keyboard.press('Tab')
  await expect(page.locator(':focus')).toBeVisible()
})
