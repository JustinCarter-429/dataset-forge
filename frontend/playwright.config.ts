import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  fullyParallel: false,
  reporter: [['list']],
  use: { baseURL: 'http://127.0.0.1:5174', screenshot: 'only-on-failure', trace: 'retain-on-failure' },
  webServer: [
    { command: 'python -c "from pathlib import Path; [p.unlink(missing_ok=True) for p in Path(\'runtime\').glob(\'playwright-carter.sqlite3*\')]" && python -m uvicorn app.main:app --host 127.0.0.1 --port 8001', cwd: '../backend', port: 8001, reuseExistingServer: false, env: { APP_ENVIRONMENT: 'test', CARTER_TEST_PROVIDER: 'deterministic', CARTER_TEST_SCENARIO: 'local_unavailable', FRONTEND_ORIGIN: 'http://127.0.0.1:5174', CARTER_KNOWLEDGE_DATABASE: 'runtime/playwright-carter.sqlite3' } },
    { command: 'npm run dev -- --host 127.0.0.1 --port 5174', port: 5174, reuseExistingServer: false, env: { VITE_API_BASE_URL: 'http://127.0.0.1:8001/api' } },
  ],
})
