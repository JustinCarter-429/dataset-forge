import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [repoArg, artifactArg] = process.argv.slice(2);
if (!repoArg || !artifactArg) {
  throw new Error("Usage: node capture_screenshots.mjs <repo-root> <artifact-dir>");
}
const repo = path.resolve(repoArg);
const artifacts = path.resolve(artifactArg);
const requireFromFrontend = createRequire(path.join(repo, "frontend", "package.json"));
const { chromium } = requireFromFrontend("playwright");
const screenshotDir = path.join(artifacts, "screenshots");
fs.mkdirSync(screenshotDir, { recursive: true });

const systemBrowserCandidates = process.platform === "win32" ? [
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
  path.join(process.env.ProgramFiles || "", "Google", "Chrome", "Application", "chrome.exe"),
  path.join(process.env["ProgramFiles(x86)"] || "", "Microsoft", "Edge", "Application", "msedge.exe"),
].filter(Boolean) : [];
const executablePath = systemBrowserCandidates.find((candidate) => fs.existsSync(candidate));
const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
const consoleErrors = [];
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => consoleErrors.push(error.message));
await page.goto(pathToFileURL(path.join(artifacts, "index.html")).href, { waitUntil: "load" });
await page.waitForSelector("#chart-comparison .plotly", { timeout: 30_000 });
await page.screenshot({ path: path.join(screenshotDir, "dashboard-desktop.png"), fullPage: true });
await page.locator("#certification-summary").screenshot({ path: path.join(screenshotDir, "certification-summary.png") });
await page.locator("#base-vs-tuned").screenshot({ path: path.join(screenshotDir, "base-versus-fine-tuned.png") });
await page.locator("#confusion-matrix").screenshot({ path: path.join(screenshotDir, "confusion-matrix.png") });
await page.locator("#dataset-types").screenshot({ path: path.join(screenshotDir, "per-dataset-type.png") });

const desktop = await page.evaluate(() => ({
  viewportWidth: innerWidth,
  documentWidth: document.documentElement.scrollWidth,
  horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1,
  badge: document.querySelector("#certification-badge")?.textContent?.trim(),
  plotCount: document.querySelectorAll(".plotly").length,
  visiblePanels: [...document.querySelectorAll(".panel")].filter((node) => {
    const box = node.getBoundingClientRect();
    return box.width > 0 && box.height > 0;
  }).length,
}));

await page.setViewportSize({ width: 390, height: 844 });
await page.reload({ waitUntil: "load" });
await page.waitForSelector("#chart-comparison .plotly", { timeout: 30_000 });
await page.screenshot({ path: path.join(screenshotDir, "dashboard-mobile.png"), fullPage: true });
const mobile = await page.evaluate(() => ({
  viewportWidth: innerWidth,
  documentWidth: document.documentElement.scrollWidth,
  horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1,
  plotCount: document.querySelectorAll(".plotly").length,
}));

const result = {
  status: !desktop.horizontalOverflow && !mobile.horizontalOverflow && desktop.plotCount === 4 && mobile.plotCount === 4 && consoleErrors.length === 0 ? "passed" : "failed",
  desktop,
  mobile,
  consoleErrors,
  screenshots: [
    "screenshots/dashboard-desktop.png",
    "screenshots/dashboard-mobile.png",
    "screenshots/certification-summary.png",
    "screenshots/base-versus-fine-tuned.png",
    "screenshots/confusion-matrix.png",
    "screenshots/per-dataset-type.png",
  ],
};
fs.writeFileSync(path.join(artifacts, "logs", "visual-verification.json"), JSON.stringify(result, null, 2) + "\n");
await browser.close();
if (result.status !== "passed") process.exitCode = 1;
