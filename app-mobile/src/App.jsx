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
  const [mode, setMode] = useState(MODES.MENU);
  const [recognizedText, setRecognizedText] = useState("Saya ingin minum");
  const [transcript, setTranscript] = useState("");
  const [speechStatus, setSpeechStatus] = useState("Siap mendengarkan");
  const [isListening, setIsListening] = useState(false);
  const [roomId, setRoomId] = useState(DEFAULT_ROOM_ID);
  const [activeRoomId, setActiveRoomId] = useState(DEFAULT_ROOM_ID);
  const [socketStatus, setSocketStatus] = useState("Menghubungkan realtime...");
  const [messages, setMessages] = useState([]);

  const speechRecognition = useMemo(() => {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }, []);

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
      setMessages((currentMessages) => [...currentMessages, message].slice(-2));

      if (message.sender === "sign") {
        setRecognizedText(message.text);
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

  const speakText = (text) => {
    if (!text) {
      return;
    }

    if (!("speechSynthesis" in window)) {
      alert("Browser belum mendukung text-to-speech.");
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "id-ID";
    utterance.rate = 0.95;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
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
          {mode === MODES.CONVERSATION && (
            <Header onBack={() => setMode(MODES.MENU)} />
          )}

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

function Header({ onBack }) {
  return (
    <header className="mb-3 flex items-center justify-between sm:mb-5">
      <button
        type="button"
        onClick={onBack}
        className="rounded-full bg-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition active:scale-95"
      >
        ← Kembali
      </button>
      <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-700">
        Mobile Web
      </span>
    </header>
  );
}

function ActivityIcon({ className }) {
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
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
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
          <ActivityIcon className="h-10 w-10 text-emerald-700" />
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

function SignToSpeechScreen({
  recognizedText,
  onSpeak,
  onSend,
  socketStatus,
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

      {/* Area stream kamera (gelap) full-bleed */}
      <div className="-mx-5 bg-slate-950 px-5 pb-6 pt-4">
        <span className="mb-3 inline-flex items-center gap-2 rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-300">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          Mendeteksi Gerakan...
        </span>
        <div className="flex min-h-52 flex-col items-center justify-center rounded-2xl border border-dashed border-slate-600 p-6 text-center">
          <CameraIcon className="h-12 w-12 text-slate-500" />
          <p className="mt-3 text-sm font-semibold text-slate-400">
            Stream Kamera Aktif
          </p>
        </div>
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
            className="mt-6 inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-teal-600 to-emerald-500 px-7 py-3 text-base font-bold text-white shadow-lg shadow-emerald-200 transition hover:brightness-105 active:scale-[0.98]"
          >
            <SpeakerIcon className="h-5 w-5" />
            Membacakan
          </button>
          <button
            type="button"
            onClick={onSend}
            className="mt-4 text-xs font-semibold text-slate-400 underline-offset-2 transition hover:text-slate-600 hover:underline"
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
    <div className="flex flex-1 flex-col gap-3">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-violet-600">
          Mode Dua Arah
        </p>
        <h1 className="mt-1 text-xl font-black text-slate-950">
          💬 Mode Percakapan
        </h1>
      </div>

      <div className="rounded-[1.5rem] bg-white p-3 shadow-sm">
        <label className="grid gap-2">
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
            Room Percakapan
          </span>
          <div className="flex gap-2">
            <input
              value={roomId}
              onChange={(event) => setRoomId(event.target.value)}
              className="min-w-0 flex-1 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-bold outline-none focus:border-violet-400 focus:ring-4 focus:ring-violet-100"
              placeholder="demo-ta"
            />
            <button
              type="button"
              onClick={onJoinRoom}
              className="rounded-2xl bg-slate-900 px-3 py-2 text-sm font-black text-white active:scale-95"
            >
              Gabung
            </button>
          </div>
        </label>
        <p className="mt-2 line-clamp-1 text-[11px] font-semibold text-slate-500">
          Room aktif: <span className="text-violet-600">{activeRoomId}</span> ·{" "}
          {socketStatus}
        </p>
      </div>

      <div className="flex flex-1 flex-col justify-center gap-3 rounded-[1.75rem] bg-slate-900 p-3">
        {messages.length === 0 && (
          <div className="flex flex-1 items-center justify-center text-center text-sm font-semibold leading-6 text-slate-400">
            Belum ada pesan realtime.
          </div>
        )}

        {messages.map((message, index) => (
          <ConversationBubble
            key={`${message.timestamp}-${index}`}
            align={message.sender === "voice" ? "right" : "left"}
            icon={message.sender === "voice" ? "🎤" : "🤟"}
            label={
              message.sender === "voice"
                ? "Orang Normal"
                : "Penyandang Disabilitas"
            }
            caption={
              message.sender === "voice"
                ? "Dari speech-to-text"
                : "Dari robot/model"
            }
            text={message.text}
            actionLabel={message.sender === "sign" ? "🔊 Bacakan" : undefined}
            onAction={() => onSpeak(message.text)}
          />
        ))}
      </div>

      <div className="grid gap-2 rounded-[1.5rem] bg-white p-3 shadow-sm">
        <label className="grid gap-2">
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
            Simulasi hasil robot/model
          </span>
          <textarea
            value={recognizedText}
            onChange={(event) => setRecognizedText(event.target.value)}
            rows={1}
            className="w-full resize-none rounded-2xl border border-slate-200 bg-slate-50 p-3 text-sm font-bold leading-5 outline-none focus:border-violet-400 focus:ring-4 focus:ring-violet-100"
            placeholder="Hasil bahasa isyarat dari server akan muncul di sini"
          />
        </label>

        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={onSendSign}
            className="rounded-2xl bg-emerald-500 px-3 py-3 text-sm font-black text-white shadow-lg shadow-emerald-100 transition hover:bg-emerald-600 active:scale-[0.98]"
          >
            🤟 Kirim Hasil
          </button>
          <button
            type="button"
            onClick={onStartListening}
            disabled={isListening}
            className="rounded-2xl bg-violet-600 px-3 py-3 text-sm font-black text-white shadow-lg shadow-violet-100 transition hover:bg-violet-700 active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
          >
            {isListening ? "Mendengar..." : "🎤 Balas"}
          </button>
        </div>
        <p className="line-clamp-2 text-center text-xs font-semibold leading-5 text-slate-500">
          {speechStatus}
        </p>
      </div>
    </div>
  );
}

function ConversationBubble({
  align,
  icon,
  label,
  caption,
  text,
  actionLabel,
  onAction,
  disabled,
}) {
  const isRight = align === "right";

  return (
    <div className={`flex ${isRight ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[88%] rounded-[1.25rem] p-3 ${isRight ? "bg-sky-500 text-white" : "bg-white text-slate-950"}`}
      >
        <div className="mb-2 flex items-center gap-2">
          <span
            className={`flex h-8 w-8 items-center justify-center rounded-full text-lg ${isRight ? "bg-white/20" : "bg-emerald-100"}`}
          >
            {icon}
          </span>
          <span>
            <span className="block text-xs font-black">{label}</span>
            <span
              className={`block text-xs font-semibold ${isRight ? "text-sky-100" : "text-slate-500"}`}
            >
              {caption}
            </span>
          </span>
        </div>
        <p className="line-clamp-3 text-base font-black leading-6">{text}</p>
        {actionLabel && (
          <button
            type="button"
            onClick={onAction}
            disabled={disabled}
            className="mt-2 rounded-full bg-emerald-500 px-3 py-2 text-xs font-black text-white transition active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  );
}

export default App;
