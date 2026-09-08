Generate ready.wav on the laptop with:

    cd Model
    python -m src.common.generate_ready_audio

Voice: id-ID-GadisNeural (Microsoft Edge online TTS via edge-tts).
Text: Halo, robot pengenalan bahasa isyarat dan ekspresi wajah sudah siap digunakan.
Format: PCM WAV mono 24 kHz, 16-bit. Generation requires internet;
robot playback uses the saved file offline. The generator does not overwrite
existing audio. The actual ready.wav is not included until generation succeeds.
