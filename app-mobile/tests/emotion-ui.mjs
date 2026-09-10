// Optional browser check: PLAYWRIGHT_MODULE can point to a temporary Playwright install.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { createServer } from "vite";

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE
  ? pathToFileURL(process.env.PLAYWRIGHT_MODULE).href : "playwright");
const output = path.join(process.env.TEMP || "/tmp", "ainex-emotion-ui-results");
await mkdir(output, { recursive: true });
let browser, vite, backend;

try {
  backend = spawn(process.execPath, ["server/index.js"], {
    cwd: new URL("../", import.meta.url), env: { ...process.env, PORT: "0", MODEL_API_KEY: "" },
    stdio: ["ignore", "pipe", "pipe"], windowsHide: true,
  });
  const base = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Backend startup timeout")), 10000);
    backend.once("error", reject);
    backend.stdout.on("data", (chunk) => {
      const match = String(chunk).match(/0\.0\.0\.0:(\d+)/);
      if (match) { clearTimeout(timer); resolve(`http://127.0.0.1:${match[1]}`); }
    });
  });
  vite = await createServer({ server: { host: "127.0.0.1", port: 0, open: false } });
  await vite.listen();
  browser = await chromium.launch({ channel: "msedge", headless: true });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    window.__spoken = [];
    window.speechSynthesis.speak = (utterance) => window.__spoken.push(utterance.text);
    window.speechSynthesis.cancel = () => {};
  });
  const post = async (body, endpoint = "emotion-result") => {
    const response = await fetch(`${base}/api/${endpoint}`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    assert.equal(response.status, 200);
  };
  const emotion = (name) => ({ status: "detected", emotions: [name], confidence: 0.92 });
  await page.goto(`${vite.resolvedUrls.local[0]}?socketUrl=${encodeURIComponent(base)}`);
  await page.getByRole("button", { name: /Bahasa Isyarat → Suara/ }).click();
  await page.getByText("Realtime aktif: demo-ta").waitFor();
  await page.getByText("Menunggu ekspresi", { exact: true }).waitFor();
  await post(emotion("happy"));
  await page.getByRole("status").filter({ hasText: "Senang" }).waitFor();
  await page.screenshot({ path: path.join(output, "sign-happy.png"), fullPage: true });
  for (const name of ["angry", "disgust", "fear", "neutral", "sad", "surprise"]) {
    await post(emotion(name));
    await page.waitForFunction((expected) => document.querySelector('.emotion-panel').style.getPropertyValue('--emotion-accent') === expected,
      { angry: "#fda4af", disgust: "#bef264", fear: "#c4b5fd", neutral: "#99f6e4", sad: "#93c5fd", surprise: "#fcd34d" }[name]);
  }
  await post({ status: "detected", emotions: ["happy", "surprise"], confidence: 0.48 });
  await page.getByText("Senang · Terkejut", { exact: true }).waitFor();
  assert.equal(await page.locator(".emotion-face").count(), 2);
  await page.screenshot({ path: path.join(output, "sign-compound.png"), fullPage: true });
  for (const [status, title] of [["no_face", "Wajah belum terlihat"], ["uncertain", "Belum yakin"], ["paused", "Pengenalan dijeda"], ["error", "Pengenalan terhenti"]]) {
    await post({ status });
    await page.getByText(title, { exact: true }).waitFor();
    assert.equal(await page.locator(".emotion-confidence").count(), 0);
  }
  assert.deepEqual(await page.evaluate(() => window.__spoken), []);
  await post(emotion("happy"));
  await page.getByText("Senang", { exact: true }).waitFor();
  await page.getByText("Menunggu hasil baru", { exact: true }).waitFor({ timeout: 6000 });
  await page.getByRole("button", { name: "Kembali" }).click();
  await page.getByRole("button", { name: /Mode Percakapan/ }).click();
  for (const text of ["Halo", "Baik", "Terima kasih", "Saya senang bertemu dengan Anda"]) {
    await post({ text }, "sign-result");
  }
  await post(emotion("happy"));
  await page.getByText("Senang", { exact: true }).waitFor();
  await page.getByText("Saya senang bertemu dengan Anda", { exact: true }).waitFor();
  assert.equal(await page.locator(".conversation-messages > div").count(), 3);
  await page.screenshot({ path: path.join(output, "conversation-mobile.png"), fullPage: true });
  for (const [width, height] of [[320, 568], [390, 844], [1280, 900]]) {
    await page.setViewportSize({ width, height });
    await page.waitForFunction(() => {
      const chat = document.querySelector('.conversation-messages');
      return Math.abs(chat.scrollHeight - chat.clientHeight - chat.scrollTop) < 2;
    });
    const size = await page.evaluate(() => ({ w: document.documentElement.scrollWidth,
      h: document.documentElement.scrollHeight, innerW: innerWidth, innerH: innerHeight }));
    assert.ok(size.w <= size.innerW && size.h <= size.innerH, JSON.stringify(size));
    await page.screenshot({ path: path.join(output, `conversation-${width}.png`), fullPage: true });
  }
  await page.emulateMedia({ reducedMotion: "reduce" });
  assert.equal(await page.locator(".emotion-face-float").first().evaluate((node) => getComputedStyle(node).animationName), "none");
  await page.locator('input[placeholder="demo-ta"]').fill("another-room");
  await page.getByRole("button", { name: "Gabung" }).click();
  await page.getByText("Realtime aktif: another-room", { exact: false }).waitFor();
  await post(emotion("sad"));
  await page.waitForTimeout(100);
  assert.equal(await page.getByText("Sedih", { exact: true }).count(), 0);
  await post({ ...emotion("neutral"), roomId: "another-room" });
  await page.getByText("Netral", { exact: true }).waitFor();
  backend.kill();
  await page.getByText("Koneksi terputus", { exact: true }).waitFor();
  assert.deepEqual(errors, []);
  console.log(`Browser checks passed. Screenshots: ${output}`);
} finally {
  await browser?.close();
  await vite?.close();
  backend?.kill();
}
