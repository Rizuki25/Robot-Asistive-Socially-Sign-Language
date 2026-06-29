# Sign Language Assistant Mobile

Aplikasi mobile berbasis web untuk workflow tugas akhir pengenalan bahasa isyarat dari robot.

## Mode aplikasi

1. **Bahasa Isyarat → Suara**
   - Input dari kamera/robot/model pengenalan bahasa isyarat.
   - Output berupa teks hasil prediksi.
   - Teks dapat dibacakan menggunakan fitur text-to-speech browser.

2. **Suara → Teks**
   - Input suara dari orang normal melalui mikrofon.
   - Output berupa teks agar dapat dibaca penyandang disabilitas.

## Rencana integrasi

- Ganti placeholder `recognizedText` di `src/App.jsx` dengan hasil prediksi dari model/robot.
- Untuk kamera, integrasi dapat menggunakan `navigator.mediaDevices.getUserMedia` atau stream dari robot.
- Untuk speech-to-text, aplikasi sudah menyiapkan Web Speech API. Dukungan browser terbaik biasanya ada di Chrome/Edge.

## Menjalankan aplikasi

```bash
npm install
npm run dev
```

Buka URL yang muncul dari Vite. Untuk uji di HP, gunakan URL network/local IP dari laptop dan pastikan perangkat berada di jaringan yang sama.
