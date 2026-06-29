import { useMemo, useState } from 'react';

const MODES = {
  MENU: 'menu',
  SIGN_TO_SPEECH: 'sign-to-speech',
  SPEECH_TO_TEXT: 'speech-to-text',
};

function App() {
  const [mode, setMode] = useState(MODES.MENU);
  const [recognizedText, setRecognizedText] = useState('Saya ingin minum');
  const [transcript, setTranscript] = useState('');
  const [speechStatus, setSpeechStatus] = useState('Siap mendengarkan');
  const [isListening, setIsListening] = useState(false);

  const speechRecognition = useMemo(() => {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }, []);

  const speakRecognizedText = () => {
    if (!('speechSynthesis' in window)) {
      alert('Browser belum mendukung text-to-speech.');
      return;
    }

    const utterance = new SpeechSynthesisUtterance(recognizedText);
    utterance.lang = 'id-ID';
    utterance.rate = 0.95;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

  const startListening = () => {
    if (!speechRecognition) {
      setSpeechStatus('Browser belum mendukung speech-to-text. Gunakan Chrome atau Edge.');
      return;
    }

    const recognition = new speechRecognition();
    recognition.lang = 'id-ID';
    recognition.interimResults = true;
    recognition.continuous = false;

    setIsListening(true);
    setSpeechStatus('Tolong tunggu sebentar...');
    setTranscript('');

    recognition.onresult = (event) => {
      const text = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join(' ')
        .trim();

      setTranscript(text);
    };

    recognition.onerror = () => {
      setSpeechStatus('Suara belum berhasil dikenali. Coba ulangi sekali lagi.');
      setIsListening(false);
    };

    recognition.onend = () => {
      setSpeechStatus('Selesai mendengarkan');
      setIsListening(false);
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
            />
          )}

          {mode === MODES.SIGN_TO_SPEECH && (
            <SignToSpeechScreen
              recognizedText={recognizedText}
              setRecognizedText={setRecognizedText}
              onSpeak={speakRecognizedText}
            />
          )}

          {mode === MODES.SPEECH_TO_TEXT && (
            <SpeechToTextScreen
              isListening={isListening}
              speechStatus={speechStatus}
              transcript={transcript}
              onStartListening={startListening}
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

function ModeSelector({ onSignToSpeech, onSpeechToText }) {
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
          Aplikasi pendamping komunikasi antara pengenalan bahasa isyarat robot dan suara manusia.
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
        <span className="block text-base font-black text-slate-950">{title}</span>
        <span className="mt-1 block text-sm leading-5 text-slate-500">{description}</span>
      </span>
    </button>
  );
}

function SignToSpeechScreen({ recognizedText, setRecognizedText, onSpeak }) {
  return (
    <div className="flex flex-1 flex-col gap-5">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-emerald-600">
          Mode Output
        </p>
        <h1 className="mt-2 text-2xl font-black text-slate-950">🤟 Bahasa Isyarat → Suara</h1>
      </div>

      <div className="flex min-h-60 flex-1 flex-col items-center justify-center rounded-[2rem] border-2 border-dashed border-slate-300 bg-slate-100 p-5 text-center">
        <div className="mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-slate-900 text-4xl text-white">
          📷
        </div>
        <p className="text-lg font-black text-slate-900">Kamera / Stream Robot</p>
        <p className="mt-2 text-sm leading-6 text-slate-500">
          Area ini disiapkan untuk kamera, stream robot, atau frame yang dikirim ke model BiLSTM.
        </p>
      </div>

      <div className="rounded-[2rem] bg-white p-5 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-[0.2em] text-slate-500">Hasil</h2>
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

        <button
          type="button"
          onClick={onSpeak}
          className="mt-4 w-full rounded-2xl bg-emerald-500 px-5 py-4 text-base font-black text-white shadow-lg shadow-emerald-200 transition hover:bg-emerald-600 active:scale-[0.98]"
        >
          🔊 Membacakan
        </button>
      </div>
    </div>
  );
}

function SpeechToTextScreen({ isListening, speechStatus, transcript, onStartListening }) {
  return (
    <div className="flex flex-1 flex-col gap-5">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-600">
          Mode Input
        </p>
        <h1 className="mt-2 text-2xl font-black text-slate-950">🎤 Suara → Teks</h1>
      </div>

      <div className="flex flex-1 flex-col items-center justify-center rounded-[2rem] bg-slate-900 p-6 text-center text-white">
        <div className={`mb-5 flex h-24 w-24 items-center justify-center rounded-full text-5xl ${isListening ? 'animate-pulse bg-red-500' : 'bg-sky-500'}`}>
          🎤
        </div>
        <button
          type="button"
          onClick={onStartListening}
          disabled={isListening}
          className="w-full rounded-2xl bg-white px-5 py-4 text-base font-black text-slate-950 transition active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
        >
          {isListening ? 'Sedang Mendengarkan...' : 'Mulai Berbicara'}
        </button>
        <p className="mt-4 text-sm font-semibold text-slate-300">{speechStatus}</p>
      </div>

      <div className="rounded-[2rem] bg-white p-5 shadow-sm">
        <h2 className="mb-3 text-sm font-bold uppercase tracking-[0.2em] text-slate-500">
          Hasil Teks
        </h2>
        <div className="min-h-32 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xl font-black leading-8 text-slate-900">
          {transcript || 'Teks dari suara akan muncul di sini.'}
        </div>
      </div>
    </div>
  );
}

export default App;
