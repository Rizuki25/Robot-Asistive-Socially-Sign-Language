export const EMOTION_LABELS = {
  angry: "Marah", disgust: "Jijik", fear: "Takut", happy: "Senang",
  neutral: "Netral", sad: "Sedih", surprise: "Terkejut",
};

export const EMOTION_TTL_MS = 4000;
const STATES = new Set(["detected", "uncertain", "no_face", "waiting", "paused", "error"]);

export function normalizeEmotion(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) return null;
  const { status, emotions = [], confidence = null, roomId = "demo-ta" } = body;
  if (!STATES.has(status) || typeof roomId !== "string" || !roomId.trim() || roomId.length > 100) return null;
  if (!Array.isArray(emotions) || emotions.length > 2
      || emotions.some((name) => typeof name !== "string" || !Object.hasOwn(EMOTION_LABELS, name))
      || new Set(emotions).size !== emotions.length) return null;
  if (confidence !== null && (typeof confidence !== "number" || !Number.isFinite(confidence)
      || confidence < 0 || confidence > 1)) return null;
  if (status === "detected" && (!emotions.length || confidence === null || confidence < 0.4)) return null;
  if (status !== "detected" && emotions.length) return null;
  return { roomId: roomId.trim(), status, emotions, confidence,
    label: emotions.map((name) => EMOTION_LABELS[name]).join(" · ") };
}
