import { useEffect, useMemo, useRef, useState } from "react";
import { io } from "socket.io-client";

const MODES = {
  MENU: "menu",
  SIGN_TO_SPEECH: "sign-to-speech",
  SPEECH_TO_TEXT: "speech-to-text",
  CONVERSATION: "conversation",
};

const DEFAULT_ROOM_ID = "demo-ta";
const SOCKET_URL_STORAGE_KEY = "sign-language-socket-url";

function getDefaultSocketUrl() {
  const { protocol, hostname } = window.location;
  const isLocalNetwork =
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname.startsWith("192.168.") ||
    hostname.startsWith("10.") ||
    /^172\.(1[6-9]|2\d|3[0-1])\./.test(hostname);

  if (isLocalNetwork) {
    return `${protocol}//${hostname}:3001`;
  }

  return "";
}

function getSocketUrl() {
  const params = new URLSearchParams(window.location.search);
  const socketUrlFromQuery = params.get("socketUrl");

  if (socketUrlFromQuery) {
    window.localStorage.setItem(SOCKET_URL_STORAGE_KEY, socketUrlFromQuery);
    return socketUrlFromQuery;
  }

  return (
    import.meta.env.VITE_SOCKET_URL ||
    window.localStorage.getItem(SOCKET_URL_STORAGE_KEY) ||
    getDefaultSocketUrl()
  );
}

const SOCKET_URL = getSocketUrl();

function App() {
  const socketRef = useRef(null);
  const recognitionRef = useRef(null);
  const modeRef = useRef(MODES.MENU);
  const autoSpeakRef = useRef(true);
  const [mode, setMode] = useState(MODES.MENU);
  const [recognizedText, setRecognizedText] = useState("");
  const [transcript, setTranscript] = useState("");
  const [speechStatus, setSpeechStatus] = useState("Siap mendengarkan");
  const [isListening, setIsListening] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(true);
  const [roomId, setRoomId] = useState(DEFAULT_ROOM_ID);
  const [activeRoomId, setActiveRoomId] = useState(DEFAULT_ROOM_ID);
  const [socketStatus, setSocketStatus] = useState("Menghubungkan realtime...");
  const [messages, setMessages] = useState([]);

  const speechRecognition = useMemo(() => {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }, []);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    autoSpeakRef.current = autoSpeak;
  }, [autoSpeak]);

  function speakText(text, silent = false) {
    const cleanText = String(text || "").trim();

    if (!cleanText) {
      return;
    }

    if (!("speechSynthesis" in window)) {
      if (!silent) {
        alert("Browser belum mendukung text-to-speech.");
      }
      return;
    }

    const spokenText = /^[a-z]$/i.test(cleanText)
      ? `Huruf ${cleanText.toUpperCase()}`
      : cleanText;
    const utterance = new SpeechSynthesisUtterance(spokenText);
    utterance.lang = "id-ID";
    utterance.rate = 0.95;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }

  useEffect(() => {
    if (!SOCKET_URL) {
      setSocketStatus(
        "URL backend Socket.IO belum diatur. Buka app dengan ?socketUrl=URL_BACKEND_TUNNEL.",
      );
      return;
    }

    const socket = io(SOCKET_URL, {
      transports: ["websocket", "polling"],
    });

    socketRef.current = socket;

    socket.on("connect", () => {
      setSocketStatus(`Realtime aktif: ${activeRoomId}`);
      socket.emit("join-room", activeRoomId);
    });

    socket.on("disconnect", () => {
      setSocketStatus("Realtime terputus");
    });

    socket.on("connect_error", () => {
      setSocketStatus(`Gagal terhubung ke ${SOCKET_URL}`);
    });

    socket.on("message", (message) => {
      setMessages((currentMessages) =>
        [...currentMessages, message].slice(-3),
      );

      if (message.sender === "sign") {
        setRecognizedText(message.text);

        if (
          autoSpeakRef.current &&
          modeRef.current === MODES.SIGN_TO_SPEECH
        ) {
          speakText(message.text, true);
        }
      }

      if (message.sender === "voice") {
        setTranscript(message.text);
      }
    });

    return () => {
      socket.disconnect();
    };
  }, [activeRoomId]);

  const joinRoom = () => {
    const nextRoomId = roomId.trim() || DEFAULT_ROOM_ID;
    setActiveRoomId(nextRoomId);
    setMessages([]);

    if (socketRef.current?.connected) {
      socketRef.current.emit("join-room", nextRoomId);
      setSocketStatus(`Realtime aktif: ${nextRoomId}`);
    }
  };

  const sendRealtimeMessage = (sender, text) => {
    const cleanText = text.trim();

    if (!cleanText) {
      return;
    }

    if (!socketRef.current?.connected) {
      setSocketStatus(
        "Realtime belum terhubung. Pastikan server Socket.IO aktif.",
      );
      return;
    }

    socketRef.current.emit("send-message", {
      roomId: activeRoomId,
      sender,
      text: cleanText,
    });
  };

  const speakRecognizedText = () => {
    speakText(recognizedText);
  };

  const startListening = (onFinalTranscript) => {
    if (!window.isSecureContext) {
      setSpeechStatus(
        "Speech-to-text di HP membutuhkan HTTPS. Buka lewat HTTPS, misalnya Cloudflare Tunnel atau deploy online.",
      );
      return;
    }

    if (!speechRecognition) {
      setSpeechStatus(
        "Browser belum mendukung speech-to-text. Gunakan Chrome atau Edge.",
      );
      return;
    }

    const recognition = new speechRecognition();
    recognitionRef.current = recognition;
    recognition.lang = "id-ID";
    recognition.interimResults = true;
    recognition.continuous = false;

    let finalTranscript = "";
    let hasRecognitionError = false;

    setIsListening(true);
    setSpeechStatus("Tolong tunggu sebentar...");
    setTranscript("");

    recognition.onresult = (event) => {
      const text = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join(" ")
        .trim();

      finalTranscript = text;
      setTranscript(text);
    };

    recognition.onerror = (event) => {
      hasRecognitionError = true;

      const errorMessages = {
        "not-allowed": "Izin mikrofon ditolak atau halaman belum HTTPS.",
        "service-not-allowed":
          "Layanan speech-to-text diblokir browser/jaringan.",
        "no-speech":
          "Tidak ada suara terdeteksi. Coba bicara lebih dekat ke mikrofon.",
        network:
          "Koneksi speech-to-text bermasalah. Coba jaringan lain atau HTTPS.",
      };

      setSpeechStatus(
        errorMessages[event.error] ||
          `Speech-to-text gagal: ${event.error || "error tidak diketahui"}`,
      );
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      recognitionRef.current = null;

      if (hasRecognitionError) {
        return;
      }

      setSpeechStatus("Selesai mendengarkan");

      if (finalTranscript && typeof onFinalTranscript === "function") {
        onFinalTranscript(finalTranscript);
      }
    };

    recognition.start();
  };

  const stopListening = () => {
    recognitionRef.current?.stop();
  };

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-4 text-slate-950 sm:py-6">
      <section className="mx-auto flex min-h-[calc(100vh-2rem)] w-full max-w-md flex-col rounded-[2rem] bg-white p-2 shadow-xl shadow-slate-300/50 ring-1 ring-slate-100 sm:min-h-[calc(100vh-3rem)]">
        <div className="flex flex-1 flex-col rounded-[1.5rem] bg-slate-50 p-5">
          {mode === MODES.MENU && (
            <ModeSelector
              onSignToSpeech={() => setMode(MODES.SIGN_TO_SPEECH)}
              onSpeechToText={() => setMode(MODES.SPEECH_TO_TEXT)}
              onConversation={() => setMode(MODES.CONVERSATION)}
            />
          )}

          {mode === MODES.SIGN_TO_SPEECH && (
            <SignToSpeechScreen
              recognizedText={recognizedText}
              onSpeak={speakRecognizedText}
              onSend={() => sendRealtimeMessage("sign", recognizedText)}
              socketStatus={socketStatus}
              autoSpeak={autoSpeak}
              onToggleAutoSpeak={() => setAutoSpeak((current) => !current)}
              activeRoomId={activeRoomId}
              onBack={() => setMode(MODES.MENU)}
            />
          )}

          {mode === MODES.SPEECH_TO_TEXT && (
            <SpeechToTextScreen
              isListening={isListening}
              speechStatus={speechStatus}
              transcript={transcript}
              socketStatus={socketStatus}
              activeRoomId={activeRoomId}
              onBack={() => setMode(MODES.MENU)}
              onStartListening={() =>
                startListening((text) => sendRealtimeMessage("voice", text))
              }
              onStopListening={stopListening}
            />
          )}

          {mode === MODES.CONVERSATION && (
            <ConversationScreen
              activeRoomId={activeRoomId}
              roomId={roomId}
              setRoomId={setRoomId}
              onJoinRoom={joinRoom}
              onBack={() => setMode(MODES.MENU)}
              socketStatus={socketStatus}
              recognizedText={recognizedText}
              setRecognizedText={setRecognizedText}
              messages={messages}
              speechStatus={speechStatus}
              isListening={isListening}
              onSpeak={speakText}
              onSendSign={() => sendRealtimeMessage("sign", recognizedText)}
              onStartListening={() =>
                startListening((text) => sendRealtimeMessage("voice", text))
              }
            />
          )}
        </div>
      </section>
    </main>
  );
}

function HandIcon({ className }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M18 11V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2" />
      <path d="M14 10V4a2 2 0 0 0-2-2 2 2 0 0 0-2 2v2" />
      <path d="M10 10.5V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2v8" />
      <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
    </svg>
  );
}

function MicIcon({ className }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M5 10v1a7 7 0 0 0 14 0v-1" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </svg>
  );
}

function ChatIcon({ className }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
    </svg>
  );
}

function ModeSelector({ onSignToSpeech, onSpeechToText, onConversation }) {
  return (
    <div className="flex flex-1 flex-col justify-center gap-8">
      <div className="text-center">
        <div className="mx-auto mb-5 flex h-24 w-24 items-center justify-center rounded-full bg-emerald-100">
          <HandIcon className="h-12 w-12 text-emerald-700" />
        </div>
        <h1 className="text-3xl font-black leading-tight tracking-tight text-slate-900">
          Sign Language Assistant
        </h1>
        <p className="mt-2 text-sm font-medium text-slate-400">
          Jembatan Komunikasi Tanpa Batas
        </p>
      </div>

      <div className="grid gap-4">
        <ModeButton
          icon={<HandIcon className="h-6 w-6" />}
          title="Bahasa Isyarat → Suara"
          description="Untuk pengguna bahasa isyarat"
          onClick={onSignToSpeech}
        />
        <ModeButton
          icon={<MicIcon className="h-6 w-6" />}
          title="Suara → Teks"
          description="Untuk pengguna suara/audio"
          onClick={onSpeechToText}
        />
        <ModeButton
          icon={<ChatIcon className="h-6 w-6" />}
          title="Mode Percakapan"
          description="Komunikasi dua arah (Demo)"
          active
          onClick={onConversation}
        />
      </div>
    </div>
  );
}

function ModeButton({ icon, title, description, onClick, active = false }) {
  if (active) {
    return (
      <button
        type="button"
        onClick={onClick}
        className="flex items-center gap-4 rounded-2xl bg-gradient-to-r from-teal-600 to-emerald-500 p-4 text-left shadow-lg shadow-emerald-200 transition hover:brightness-105 active:scale-[0.98]"
      >
        <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-white/20 text-white">
          {icon}
        </span>
        <span>
          <span className="block text-base font-bold text-white">{title}</span>
          <span className="mt-0.5 block text-sm font-medium text-emerald-50">
            {description}
          </span>
        </span>
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-4 rounded-2xl border border-slate-100 bg-white p-4 text-left shadow-sm transition hover:border-emerald-200 hover:shadow-md active:scale-[0.98]"
    >
      <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
        {icon}
      </span>
      <span>
        <span className="block text-base font-bold text-slate-900">
          {title}
        </span>
        <span className="mt-0.5 block text-sm font-medium text-slate-400">
          {description}
        </span>
      </span>
    </button>
  );
}

function BackArrowIcon({ className }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <line x1="19" y1="12" x2="5" y2="12" />
      <polyline points="12 19 5 12 12 5" />
    </svg>
  );
}

function CameraIcon({ className }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  );
}

function SpeakerIcon({ className }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
    </svg>
  );
}

function SendIcon({ className }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function SignToSpeechScreen({
  recognizedText,
  onSpeak,
  onSend,
  socketStatus,
  autoSpeak,
  onToggleAutoSpeak,
  activeRoomId,
  onBack,
}) {
  return (
    <div className="-mt-5 flex flex-1 flex-col">
      {/* Header hijau full-bleed */}
      <div className="-mx-5 rounded-t-[1.5rem] bg-gradient-to-r from-teal-600 to-emerald-500 px-5 py-4">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-white/95 px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-white active:scale-95"
          >
            <BackArrowIcon className="h-4 w-4" />
            Kembali
          </button>
          <h1 className="text-base font-bold text-white">
            Bahasa Isyarat → Suara
          </h1>
        </div>
      </div>

      {/* Area kamera dipertahankan, stream video dinonaktifkan */}
      <div className="-mx-5 bg-slate-950 px-5 pb-6 pt-4">
        <span className="mb-3 inline-flex items-center gap-2 rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-300">
          <span className="h-2 w-2 rounded-full bg-amber-400" />
          Kamera Nonaktif
        </span>
        <div className="flex min-h-52 w-full min-w-0 max-w-full flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed border-slate-600 p-6 text-center">
          <CameraIcon className="h-12 w-12 text-slate-500" />
          <p className="mt-3 max-w-full break-words text-sm font-semibold text-slate-400">
            Stream Kamera Dinonaktifkan
          </p>
        </div>
        <p className="mt-3 text-center text-[11px] font-medium text-slate-400">
          Hanya hasil teks yang diterima dari model
        </p>
        <p className="mt-3 flex items-center justify-center gap-1.5 text-center text-[11px] font-medium text-slate-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
          Room: <span className="font-semibold text-emerald-300">
            {activeRoomId}
          </span>
          <span className="text-slate-600">·</span>
          <span className="truncate">{socketStatus}</span>
        </p>
      </div>

      {/* Hasil terjemahan */}
      <div className="flex flex-1 flex-col pt-5">
        <p className="text-xs font-bold uppercase tracking-[0.15em] text-teal-700/70">
          Hasil Terjemahan
        </p>
        <div className="mt-3 flex flex-1 flex-col items-center justify-center rounded-3xl bg-white p-6 text-center shadow-sm">
          <p className="text-3xl font leading-snug text-slate-900">
            &ldquo;{recognizedText || "..."}&rdquo;
          </p>
          <button
            type="button"
            onClick={onSpeak}
            disabled={!recognizedText}
            className="mt-6 inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-teal-600 to-emerald-500 px-7 py-3 text-base font-bold text-white shadow-lg shadow-emerald-200 transition hover:brightness-105 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <SpeakerIcon className="h-5 w-5" />
            Membacakan
          </button>
          <button
            type="button"
            role="switch"
            aria-checked={autoSpeak}
            onClick={onToggleAutoSpeak}
            className={`mt-4 inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-bold transition active:scale-95 ${
              autoSpeak
                ? "bg-emerald-100 text-emerald-700"
                : "bg-slate-100 text-slate-500"
            }`}
          >
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                autoSpeak ? "bg-emerald-500" : "bg-slate-400"
              }`}
            />
            Suara otomatis: {autoSpeak ? "Aktif" : "Nonaktif"}
          </button>
          <button
            type="button"
            onClick={onSend}
            disabled={!recognizedText}
            className="mt-4 text-xs font-semibold text-slate-400 underline-offset-2 transition hover:text-slate-600 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
          >
            Kirim ke semua perangkat
          </button>
        </div>
      </div>
    </div>
  );
}

function ListeningWave() {
  const bars = [0, 0.15, 0.3, 0.45, 0.15];

  return (
    <div className="flex h-10 items-center justify-center gap-1.5">
      {bars.map((delay, index) => (
        <span
          key={index}
          className="stt-wave-bar h-8 w-2 rounded-full bg-emerald-300"
          style={{ animationDelay: `${delay}s` }}
        />
      ))}
    </div>
  );
}

function SpeechToTextScreen({
  isListening,
  speechStatus,
  transcript,
  socketStatus,
  activeRoomId,
  onBack,
  onStartListening,
  onStopListening,
}) {
  return (
    <div className="-mt-5 flex flex-1 flex-col">
      {/* Header hijau full-bleed */}
      <div className="-mx-5 rounded-t-[1.5rem] bg-gradient-to-r from-teal-600 to-emerald-500 px-5 py-4">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="flex h-9 w-9 items-center justify-center rounded-full text-white transition hover:bg-white/15 active:scale-95"
            aria-label="Kembali"
          >
            <BackArrowIcon className="h-5 w-5" />
          </button>
          <h1 className="text-lg font-bold text-white">Suara → Teks</h1>
        </div>
      </div>

      {/* Isi */}
      <div className="flex flex-1 flex-col items-center justify-center gap-4 py-6">
        {isListening && (
          <p className="flex items-center gap-2 text-sm font-semibold text-emerald-600">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
            Mendengarkan...
          </p>
        )}

        <div className="flex min-h-40 w-full items-center justify-center rounded-3xl bg-white p-6 text-center shadow-sm">
          {isListening ? (
            transcript ? (
              <p className="text-xl font-bold leading-8 text-slate-900">
                {transcript}
              </p>
            ) : (
              <ListeningWave />
            )
          ) : transcript ? (
            <p className="text-xl font-bold leading-8 text-slate-900">
              {transcript}
            </p>
          ) : (
            <p className="text-sm font-medium leading-6 text-slate-400">
              Tekan tombol di bawah untuk mulai berbicara.
            </p>
          )}
        </div>
      </div>

      {/* Tombol mic bawah */}
      <div className="flex flex-col items-center gap-2 pb-6">
        <div className="relative flex h-24 w-24 items-center justify-center">
          {isListening && (
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-300/50" />
          )}
          <button
            type="button"
            onClick={isListening ? onStopListening : onStartListening}
            className={`relative flex h-24 w-24 flex-col items-center justify-center gap-1 rounded-full font-bold shadow-lg transition active:scale-95 ${
              isListening
                ? "border-2 border-red-500 bg-red-50 text-red-600 shadow-red-200"
                : "bg-gradient-to-br from-teal-600 to-emerald-500 text-white shadow-emerald-200"
            }`}
          >
            <MicIcon className="h-7 w-7" />
            <span className="text-xs">
              {isListening ? "Stop" : "Berbicara"}
            </span>
          </button>
        </div>
        <p className="max-w-[16rem] text-center text-[11px] font-medium text-slate-400">
          Room: <span className="font-semibold text-emerald-600">
            {activeRoomId}
          </span>{" "}
          · {isListening ? speechStatus : socketStatus}
        </p>
      </div>
    </div>
  );
}

function ConversationScreen({
  activeRoomId,
  roomId,
  setRoomId,
  onJoinRoom,
  onBack,
  socketStatus,
  recognizedText,
  setRecognizedText,
  messages,
  speechStatus,
  isListening,
  onSpeak,
  onSendSign,
  onStartListening,
}) {
  return (
    <div className="-mt-5 flex flex-1 flex-col">
      {/* Header hijau full-bleed */}
      <div className="-mx-5 rounded-t-[1.5rem] bg-gradient-to-r from-teal-600 to-emerald-500 px-5 py-4">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="flex h-9 w-9 items-center justify-center rounded-full text-white transition hover:bg-white/15 active:scale-95"
            aria-label="Kembali"
          >
            <BackArrowIcon className="h-5 w-5" />
          </button>
          <h1 className="text-lg font-bold text-white">Mode Percakapan</h1>
        </div>
      </div>

      {/* Bar room server (nama room tetap dimunculkan) */}
      <div className="-mx-5 border-b border-slate-100 bg-white px-5 py-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold uppercase tracking-wide text-slate-400">
            Room
          </span>
          <input
            value={roomId}
            onChange={(event) => setRoomId(event.target.value)}
            className="min-w-0 flex-1 rounded-lg border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-semibold outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
            placeholder="demo-ta"
          />
          <button
            type="button"
            onClick={onJoinRoom}
            className="rounded-lg bg-slate-800 px-3 py-1 text-xs font-bold text-white active:scale-95"
          >
            Gabung
          </button>
        </div>
        <p className="mt-1 truncate text-[10px] font-medium text-slate-400">
          Aktif:{" "}
          <span className="font-semibold text-emerald-600">{activeRoomId}</span>{" "}
          · {socketStatus}
        </p>
      </div>

      {/* Area chat */}
      <div
        className="-mx-5 flex min-h-0 flex-1 flex-col gap-4 overflow-hidden px-5 py-4"
      >
        {messages.length === 0 && (
          <div className="flex flex-1 items-center justify-center text-center text-sm font-medium leading-6 text-slate-400">
            Belum ada pesan realtime.
          </div>
        )}

        {messages.map((message, index) => (
          <ConversationBubble
            key={`${message.timestamp}-${index}`}
            align={message.sender === "voice" ? "right" : "left"}
            label={
              message.sender === "voice"
                ? "Anda (Suara)"
                : "Penyandang Disabilitas (Isyarat)"
            }
            text={message.text}
            showAction={message.sender === "sign"}
            onAction={() => onSpeak(message.text)}
          />
        ))}
      </div>

      {/* Panel input bawah */}
      <div className="-mx-5 -mb-5 flex flex-col gap-3 rounded-b-[1.5rem] border-t border-slate-100 bg-white px-5 pb-5 pt-4">
        <button
          type="button"
          onClick={onStartListening}
          disabled={isListening}
          className="mx-auto inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-teal-600 to-emerald-500 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-emerald-200 transition active:scale-95 disabled:opacity-60"
        >
          <MicIcon className="h-5 w-5" />
          {isListening ? "Mendengarkan..." : "Balas dengan Suara"}
        </button>

        <div className="rounded-2xl bg-slate-50 p-3">
          <p className="mb-2 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.15em] text-slate-500">
            🔧 Simulasi Input Isyarat (Robot/Server)
          </p>
          <div className="flex items-center gap-2">
            <input
              value={recognizedText}
              onChange={(event) => setRecognizedText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  onSendSign();
                }
              }}
              className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
              placeholder="Ketik simulasi isyarat..."
            />
            <button
              type="button"
              onClick={onSendSign}
              aria-label="Kirim simulasi isyarat"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-slate-500 text-white transition hover:bg-slate-600 active:scale-95"
            >
              <SendIcon className="h-5 w-5" />
            </button>
          </div>
        </div>

        {speechStatus && (
          <p className="line-clamp-2 text-center text-[11px] font-medium leading-4 text-slate-400">
            {speechStatus}
          </p>
        )}
      </div>
    </div>
  );
}

function ConversationBubble({ align, label, text, showAction, onAction }) {
  const isRight = align === "right";

  return (
    <div className={`flex flex-col ${isRight ? "items-end" : "items-start"}`}>
      <span className="mb-1 px-1 text-[11px] font-semibold text-slate-400">
        {label}
      </span>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 shadow-sm ${
          isRight
            ? "bg-gradient-to-br from-teal-600 to-emerald-500 text-white"
            : "bg-white text-slate-900"
        }`}
      >
        <p className="text-sm font-semibold leading-6">{text}</p>
        {showAction && (
          <button
            type="button"
            onClick={onAction}
            className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 transition active:scale-95"
          >
            <SpeakerIcon className="h-3.5 w-3.5" />
            Bacakan
          </button>
        )}
      </div>
    </div>
  );
}

export default App;
