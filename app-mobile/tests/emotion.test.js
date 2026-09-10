import assert from "node:assert/strict";
import { before, after, test } from "node:test";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { io } from "socket.io-client";
import { normalizeEmotion } from "../shared/emotion.js";

let backend, base;
const clients = [];
const payload = { roomId: "test-a", status: "detected", emotions: ["happy"], confidence: 0.9 };

before(async () => {
  backend = spawn(process.execPath, ["server/index.js"], {
    cwd: new URL("../", import.meta.url),
    env: { ...process.env, PORT: "0", MODEL_API_KEY: "emotion-test-key" },
    stdio: ["ignore", "pipe", "pipe"], windowsHide: true,
  });
  base = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Backend startup timeout")), 10000);
    backend.once("error", reject);
    backend.stdout.on("data", (chunk) => {
      const match = String(chunk).match(/0\.0\.0\.0:(\d+)/);
      if (match) { clearTimeout(timer); resolve(`http://127.0.0.1:${match[1]}`); }
    });
  });
});

after(() => { clients.forEach((client) => client.disconnect()); backend?.kill(); });

async function connect(room) {
  const socket = io(base, { transports: ["websocket"], forceNew: true });
  clients.push(socket);
  await once(socket, "connect");
  await socket.timeout(2000).emitWithAck("join-room", room);
  return socket;
}

function post(body, key = "emotion-test-key") {
  return fetch(base + "/api/emotion-result", { method: "POST",
    headers: { "Content-Type": "application/json", "x-model-api-key": key }, body: JSON.stringify(body) });
}

test("validates model statuses, class names and confidence", () => {
  assert.equal(normalizeEmotion(payload).label, "Senang");
  assert.equal(normalizeEmotion({ ...payload, emotions: ["happy", "surprise"] }).label, "Senang · Terkejut");
  for (const invalid of [null, {}, { ...payload, confidence: 2 }, { ...payload, confidence: 0.2 },
    { ...payload, emotions: ["fake"] }, { ...payload, status: "no_face" }, { ...payload, roomId: [] }]) {
    assert.equal(normalizeEmotion(invalid), null);
  }
  assert.equal(normalizeEmotion({ status: "no_face" }).label, "");
});

test("endpoint enforces API key and rejects invalid input", async () => {
  assert.equal((await post(payload, "wrong")).status, 401);
  assert.equal((await post({ ...payload, confidence: "90%" })).status, 400);
});

test("emotion is room scoped, never a chat event, and resets on face loss", async () => {
  const a = await connect("test-a");
  const b = await connect("test-b");
  const otherEvents = [], chats = [];
  b.on("emotion-result", (event) => otherEvents.push(event));
  a.on("message", (event) => chats.push(event));
  const next = once(a, "emotion-result");
  assert.equal((await post(payload)).status, 200);
  assert.equal((await next)[0].label, "Senang");
  const noFace = once(a, "emotion-result");
  await post({ roomId: "test-a", status: "no_face" });
  assert.deepEqual((await noFace)[0].emotions, []);
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.deepEqual(otherEvents, []);
  assert.deepEqual(chats, []);
});

test("joining a new room leaves the previous room and replays only fresh results", async () => {
  await post({ ...payload, roomId: "test-replay" });
  const socket = await connect("test-origin");
  const replay = once(socket, "emotion-result");
  await socket.timeout(2000).emitWithAck("join-room", "test-replay");
  const [event] = await replay;
  assert.equal(event.label, "Senang");
  assert.ok(event.expiresInMs > 0 && event.expiresInMs <= 4000);
  await socket.timeout(2000).emitWithAck("join-room", "test-new");
  const oldEvents = [];
  socket.on("emotion-result", (event) => oldEvents.push(event));
  await post({ ...payload, roomId: "test-replay" });
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.deepEqual(oldEvents, []);
});
