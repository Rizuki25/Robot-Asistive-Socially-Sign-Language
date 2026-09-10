import { EMOTION_LABELS } from "../shared/emotion";

const MOODS = {
  happy: { color: "#34d399", detail: "Senyum terdeteksi pada wajah.", mouth: "M33 49 Q44 64 55 49", brows: "M27 30 Q32 26 37 29 M51 29 Q56 26 61 30" },
  sad: { color: "#93c5fd", detail: "Ekspresi sedih terdeteksi.", mouth: "M34 57 Q44 46 54 57", brows: "M27 30 L37 26 M51 26 L61 30" },
  angry: { color: "#fda4af", detail: "Ekspresi marah terdeteksi.", mouth: "M34 56 Q44 49 54 56", brows: "M27 26 L37 31 M51 31 L61 26" },
  fear: { color: "#c4b5fd", detail: "Ekspresi takut terdeteksi.", mouth: "M35 53 Q44 47 53 53 L53 57 Q44 54 35 57 Z", brows: "M27 29 Q32 23 37 27 M51 27 Q56 23 61 29" },
  disgust: { color: "#bef264", detail: "Ekspresi jijik terdeteksi.", mouth: "M34 53 Q40 49 45 54 Q50 58 55 51", brows: "M27 29 L37 31 M51 28 L61 25" },
  surprise: { color: "#fcd34d", detail: "Ekspresi terkejut terdeteksi.", mouth: "M39 51 C39 44 49 44 49 51 C49 61 39 61 39 51 Z", brows: "M27 25 Q32 20 37 25 M51 25 Q56 20 61 25" },
  neutral: { color: "#99f6e4", detail: "Ekspresi wajah tampak netral.", mouth: "M35 53 L53 53", brows: "M27 28 L37 28 M51 28 L61 28" },
};

const EMPTY_STATES = {
  waiting: ["Menunggu ekspresi", "Hasil wajah akan muncul di sini."],
  no_face: ["Wajah belum terlihat", "Posisikan wajah ke arah kamera robot."],
  uncertain: ["Belum yakin", "Ekspresi belum terbaca dengan jelas."],
  paused: ["Pengenalan dijeda", "Menunggu robot selesai merespons."],
  error: ["Pengenalan terhenti", "Menunggu pemrosesan wajah pulih."],
  stale: ["Menunggu hasil baru", "Belum ada pembaruan dari kamera."],
  disconnected: ["Koneksi terputus", "Menunggu koneksi tersambung kembali."],
};

function AnimatedFace({ emotion, secondary = false }) {
  const mood = MOODS[emotion] || MOODS.neutral;
  return (
    <svg viewBox="0 0 88 88" className={`emotion-face ${emotion ? "is-active" : "is-idle"} ${secondary ? "is-secondary" : ""}`}
      aria-hidden="true" style={{ "--face-color": emotion ? mood.color : "#cbd5e1" }}>
      <ellipse cx="44" cy="79" rx="23" ry="4" fill="currentColor" opacity=".12" />
      <g className="emotion-face-float">
        <path d="M44 18 V10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <circle cx="44" cy="8" r="4" fill="var(--face-color)" />
        <rect x="7" y="34" width="10" height="20" rx="5" fill="var(--face-color)" opacity=".7" />
        <rect x="71" y="34" width="10" height="20" rx="5" fill="var(--face-color)" opacity=".7" />
        <rect x="14" y="17" width="60" height="54" rx="22" fill="var(--face-color)" />
        <rect x="20" y="23" width="48" height="41" rx="16" fill="#103c3a" />
        <g stroke="var(--face-color)" strokeWidth="2.5" strokeLinecap="round" fill="none">
          <path d={mood.brows} />
          <g className="emotion-eyes">
            <path d="M32 36 V40 M56 36 V40" strokeWidth="4" />
          </g>
          <path d={mood.mouth} strokeLinejoin="round" />
        </g>
        {emotion === "happy" && <g fill="#fda4af" opacity=".65"><ellipse cx="27" cy="46" rx="4" ry="2" /><ellipse cx="61" cy="46" rx="4" ry="2" /></g>}
      </g>
    </svg>
  );
}

export default function EmotionPanel({ result, compact = false }) {
  const detected = result?.status === "detected";
  const emotions = detected ? result.emotions : [];
  const primary = emotions[0];
  const isCompound = emotions.length > 1;
  const [title, detail] = detected
    ? [emotions.map((name) => EMOTION_LABELS[name]).join(" · "), isCompound ? "Dua ekspresi terbaca bersamaan." : MOODS[primary].detail]
    : EMPTY_STATES[result?.status] || EMPTY_STATES.waiting;
  const confidence = detected ? Math.round(result.confidence * 100) : null;

  return (
    <section className={`emotion-panel ${compact ? "emotion-panel-compact" : "emotion-panel-hero"}`}
      aria-label="Hasil pengenalan ekspresi wajah" data-emotion-state={result?.status || "waiting"}
      style={{ "--emotion-accent": MOODS[primary]?.color || "#99f6e4" }}>
      {!compact && <div className="emotion-panel-top"><span>EKSPRESI WAJAH</span><span className={`emotion-state-dot ${detected ? "is-live" : ""}`} /> <span>{detected ? "Terdeteksi" : "Menunggu"}</span></div>}
      <div className="emotion-panel-body">
        <div className={`emotion-avatar ${isCompound ? "is-compound" : ""}`}>
          <AnimatedFace emotion={primary} />
          {isCompound && <AnimatedFace emotion={emotions[1]} secondary />}
        </div>
        <div className="emotion-copy">
          {compact && <p className="emotion-eyebrow">EKSPRESI WAJAH</p>}
          <p className="emotion-title" role="status" aria-live="polite" aria-atomic="true">{title}</p>
          <p className="emotion-detail">{detail}</p>
          {detected && <div className="emotion-confidence"><span>Keyakinan model</span><strong>{confidence}%</strong></div>}
          {!compact && detected && <div className="emotion-meter" aria-hidden="true"><span style={{ width: `${confidence}%` }} /></div>}
        </div>
      </div>
      {!compact && <p className="emotion-footnote">Dibaca dari ekspresi wajah di kamera robot</p>}
    </section>
  );
}
