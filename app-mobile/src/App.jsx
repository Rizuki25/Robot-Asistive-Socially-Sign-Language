import { useEffect, useMemo, useRef, useState } from "react";
import { io } from "socket.io-client";

const MODES = {
  MENU: "menu",
  SIGN_TO_SPEECH: "sign-to-speech",
  SPEECH_TO_TEXT: "speech-to-text",
  CONVERSATION: "conversation",
};

const DEFAULT_ROOM_ID = "demo-ta";
const SOCKET_URL =
  import.meta.env.VITE_SOCKET_URL ||
  `${window.location.protocol}//${window.location.hostname}:3001`;

function App() {
  const socketRef = useRef(null);
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
      setMessages((currentMessages) => [...currentMessages, message]);

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
    if (!speechRecognition) {
      setSpeechStatus(
        "Browser belum mendukung speech-to-text. Gunakan Chrome atau Edge.",
      );
      return;
    }

    const recognition = new speechRecognition();
    recognition.lang = "id-ID";
    recognition.interimResults = true;
    recognition.continuous = false;

    let finalTranscript = "";

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

    recognition.onerror = () => {
      setSpeechStatus(
        "Suara belum berhasil dikenali. Coba ulangi sekali lagi.",
      );
      setIsListening(false);
    };

    recognition.onend = () => {
      setSpeechStatus("Selesai mendengarkan");
      setIsListening(false);

      if (finalTranscript && typeof onFinalTranscript === "function") {
        onFinalTranscript(finalTranscript);
      }
    };

    recognition.start();
  };

  return (
    <main className="min-h-screen bg-slate-200 px-4 py-6 text-slate-950">
      <section className="mx-auto flex min-h-[calc(100vh-3rem)] w-full max-w-md flex-col rounded-[2rem] bg-slate-950 p-3 shadow-2xl">
        <div className="flex flex-1 flex-col rounded-[1.5rem] bg-slate-50 p-5">
          {mode !== MODES.MENU && <Header onBack={() => setMode(MODES.MENU)} />}

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
              setRecognizedText={setRecognizedText}
              onSpeak={speakRecognizedText}
              onSend={() => sendRealtimeMessage("sign", recognizedText)}
              socketStatus={socketStatus}
            />
          )}

          {mode === MODES.SPEECH_TO_TEXT && (
            <SpeechToTextScreen
              isListening={isListening}
              speechStatus={speechStatus}
              transcript={transcript}
              socketStatus={socketStatus}
              onStartListening={() =>
                startListening((text) => sendRealtimeMessage("voice", text))
              }
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
    <header className="mb-5 flex items-center justify-between">
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

function ModeSelector({ onSignToSpeech, onSpeechToText, onConversation }) {
  return (
    <div className="flex flex-1 flex-col justify-center gap-8">
      <div className="text-center">
        <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-3xl bg-emerald-500 text-4xl shadow-lg shadow-emerald-200">
          🤟
        </div>
        <p className="text-sm font-semibold uppercase tracking-[0.25em] text-emerald-600">
          Pilih Mode
        </p>
        <h1 className="mt-3 text-3xl font-black leading-tight text-slate-950">
          Sign Language Assistant
        </h1>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          Aplikasi pendamping komunikasi antara pengenalan bahasa isyarat robot
          dan suara manusia.
        </p>
      </div>

      <div className="grid gap-4">
        <ModeButton
          icon="🤟"
          title="Bahasa Isyarat → Suara"
          description="Hasil dari robot/model ditampilkan sebagai teks dan dibacakan."
          onClick={onSignToSpeech}
        />
        <ModeButton
          icon="🎤"
          title="Suara → Teks"
          description="Ucapan orang normal diubah menjadi teks untuk dibaca."
          onClick={onSpeechToText}
        />
        <ModeButton
          icon="💬"
          title="Mode Percakapan"
          description="Komunikasi dua arah realtime di banyak HP sekaligus."
          onClick={onConversation}
        />
      </div>
    </div>
  );
}

function ModeButton({ icon, title, description, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-4 rounded-3xl border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-emerald-300 hover:shadow-md active:scale-[0.98]"
    >
      <span className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-slate-100 text-3xl">
        {icon}
      </span>
      <span>
        <span className="block text-base font-black text-slate-950">
          {title}
        </span>
        <span className="mt-1 block text-sm leading-5 text-slate-500">
          {description}
        </span>
      </span>
    </button>
  );
}

function SignToSpeechScreen({
  recognizedText,
  setRecognizedText,
  onSpeak,
  onSend,
  socketStatus,
}) {
  return (
    <div className="flex flex-1 flex-col gap-5">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-emerald-600">
          Mode Output
        </p>
        <h1 className="mt-2 text-2xl font-black text-slate-950">
          🤟 Bahasa Isyarat → Suara
        </h1>
        <p className="mt-2 text-xs font-semibold text-slate-500">
          {socketStatus}
        </p>
      </div>

      <div className="flex min-h-60 flex-1 flex-col items-center justify-center rounded-[2rem] border-2 border-dashed border-slate-300 bg-slate-100 p-5 text-center">
        <div className="mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-slate-900 text-4xl text-white">
          📷
        </div>
        <p className="text-lg font-black text-slate-900">
          Kamera / Stream Robot
        </p>
        <p className="mt-2 text-sm leading-6 text-slate-500">
          Area ini disiapkan untuk kamera, stream robot, atau frame yang dikirim
          ke model BiLSTM.
        </p>
      </div>

      <div className="rounded-[2rem] bg-white p-5 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-[0.2em] text-slate-500">
            Hasil
          </h2>
          <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-700">
            Prediksi
          </span>
        </div>

        <textarea
          value={recognizedText}
          onChange={(event) => setRecognizedText(event.target.value)}
          rows={3}
          className="w-full resize-none rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xl font-black leading-8 outline-none focus:border-emerald-400 focus:ring-4 focus:ring-emerald-100"
          placeholder="Hasil pengenalan bahasa isyarat akan muncul di sini"
        />

        <div className="mt-4 grid gap-3">
          <button
            type="button"
            onClick={onSpeak}
            className="w-full rounded-2xl bg-emerald-500 px-5 py-4 text-base font-black text-white shadow-lg shadow-emerald-200 transition hover:bg-emerald-600 active:scale-[0.98]"
          >
            🔊 Membacakan
          </button>
          <button
            type="button"
            onClick={onSend}
            className="w-full rounded-2xl bg-slate-900 px-5 py-4 text-base font-black text-white transition hover:bg-slate-800 active:scale-[0.98]"
          >
            Kirim ke Semua Perangkat
          </button>
        </div>
      </div>
    </div>
  );
}

function SpeechToTextScreen({
  isListening,
  speechStatus,
  transcript,
  socketStatus,
  onStartListening,
}) {
  return (
    <div className="flex flex-1 flex-col gap-5">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-600">
          Mode Input
        </p>
        <h1 className="mt-2 text-2xl font-black text-slate-950">
          🎤 Suara → Teks
        </h1>
        <p className="mt-2 text-xs font-semibold text-slate-500">
          {socketStatus}
        </p>
      </div>

      <div className="flex flex-1 flex-col items-center justify-center rounded-[2rem] bg-slate-900 p-6 text-center text-white">
        <div
          className={`mb-5 flex h-24 w-24 items-center justify-center rounded-full text-5xl ${isListening ? "animate-pulse bg-red-500" : "bg-sky-500"}`}
        >
          🎤
        </div>
        <button
          type="button"
          onClick={onStartListening}
          disabled={isListening}
          className="w-full rounded-2xl bg-white px-5 py-4 text-base font-black text-slate-950 transition active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
        >
          {isListening ? "Sedang Mendengarkan..." : "Mulai Berbicara"}
        </button>
        <p className="mt-4 text-sm font-semibold text-slate-300">
          {speechStatus}
        </p>
      </div>

      <div className="rounded-[2rem] bg-white p-5 shadow-sm">
        <h2 className="mb-3 text-sm font-bold uppercase tracking-[0.2em] text-slate-500">
          Hasil Teks
        </h2>
        <div className="min-h-32 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xl font-black leading-8 text-slate-900">
          {transcript || "Teks dari suara akan muncul di sini."}
        </div>
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
    <div className="flex flex-1 flex-col gap-5">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-violet-600">
          Mode Dua Arah
        </p>
        <h1 className="mt-2 text-2xl font-black text-slate-950">
          💬 Mode Percakapan
        </h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Semua HP/laptop yang masuk room yang sama akan menerima chat realtime.
        </p>
      </div>

      <div className="rounded-[2rem] bg-white p-4 shadow-sm">
        <label className="grid gap-2">
          <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">
            Room Percakapan
          </span>
          <div className="flex gap-2">
            <input
              value={roomId}
              onChange={(event) => setRoomId(event.target.value)}
              className="min-w-0 flex-1 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold outline-none focus:border-violet-400 focus:ring-4 focus:ring-violet-100"
              placeholder="demo-ta"
            />
            <button
              type="button"
              onClick={onJoinRoom}
              className="rounded-2xl bg-slate-900 px-4 py-3 text-sm font-black text-white active:scale-95"
            >
              Gabung
            </button>
          </div>
        </label>
        <p className="mt-2 text-xs font-semibold text-slate-500">
          Room aktif: <span className="text-violet-600">{activeRoomId}</span> ·{" "}
          {socketStatus}
        </p>
      </div>

      <div className="flex min-h-80 flex-1 flex-col gap-4 overflow-y-auto rounded-[2rem] bg-slate-900 p-4">
        {messages.length === 0 && (
          <div className="flex flex-1 items-center justify-center text-center text-sm font-semibold leading-6 text-slate-400">
            Belum ada pesan realtime. Kirim hasil robot/model atau tekan tombol
            balas dengan suara.
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

      <div className="grid gap-3 rounded-[2rem] bg-white p-4 shadow-sm">
        <label className="grid gap-2">
          <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">
            Simulasi hasil robot/model
          </span>
          <textarea
            value={recognizedText}
            onChange={(event) => setRecognizedText(event.target.value)}
            rows={2}
            className="w-full resize-none rounded-2xl border border-slate-200 bg-slate-50 p-3 text-base font-bold leading-6 outline-none focus:border-violet-400 focus:ring-4 focus:ring-violet-100"
            placeholder="Hasil bahasa isyarat dari server akan muncul di sini"
          />
        </label>

        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={onSendSign}
            className="rounded-2xl bg-emerald-500 px-4 py-4 text-sm font-black text-white shadow-lg shadow-emerald-100 transition hover:bg-emerald-600 active:scale-[0.98]"
          >
            🤟 Kirim Hasil
          </button>
          <button
            type="button"
            onClick={onStartListening}
            disabled={isListening}
            className="rounded-2xl bg-violet-600 px-4 py-4 text-sm font-black text-white shadow-lg shadow-violet-100 transition hover:bg-violet-700 active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
          >
            {isListening ? "Mendengar..." : "🎤 Balas"}
          </button>
        </div>
        <p className="text-center text-sm font-semibold text-slate-500">
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
        className={`max-w-[86%] rounded-[1.5rem] p-4 ${isRight ? "bg-sky-500 text-white" : "bg-white text-slate-950"}`}
      >
        <div className="mb-2 flex items-center gap-2">
          <span
            className={`flex h-9 w-9 items-center justify-center rounded-full text-xl ${isRight ? "bg-white/20" : "bg-emerald-100"}`}
          >
            {icon}
          </span>
          <span>
            <span className="block text-sm font-black">{label}</span>
            <span
              className={`block text-xs font-semibold ${isRight ? "text-sky-100" : "text-slate-500"}`}
            >
              {caption}
            </span>
          </span>
        </div>
        <p className="text-lg font-black leading-7">{text}</p>
        {actionLabel && (
          <button
            type="button"
            onClick={onAction}
            disabled={disabled}
            className="mt-3 rounded-full bg-emerald-500 px-4 py-2 text-sm font-black text-white transition active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  );
}

export default App;
