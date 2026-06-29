# Sign Language Assistant Mobile

Aplikasi mobile berbasis web untuk workflow tugas akhir pengenalan bahasa isyarat dari robot.

## Mode aplikasi

1. **Bahasa Isyarat → Suara**
   - Input dari kamera/robot/model pengenalan bahasa isyarat.
   - Server memproses data menggunakan model pengenalan.
   - Output berupa teks hasil prediksi.
   - Teks dapat dibacakan menggunakan fitur text-to-speech browser.

2. **Suara → Teks**
   - Input suara dari orang normal melalui mikrofon.
   - Output berupa teks agar dapat dibaca penyandang disabilitas.

3. **Mode Percakapan**
   - Menggabungkan dua arah komunikasi dalam satu layar.
   - Hasil bahasa isyarat dari robot/model tampil sebagai pesan penyandang disabilitas.
   - Balasan suara orang normal diproses menjadi teks dan tampil sebagai pesan balasan.
   - Cocok digunakan sebagai mode utama saat demo komunikasi dua arah.

## Rencana integrasi

- Ganti placeholder `recognizedText` di `src/App.jsx` dengan hasil prediksi dari model/robot.
- Untuk kamera, integrasi dapat menggunakan `navigator.mediaDevices.getUserMedia` atau stream dari robot.
- Untuk server/model, frontend dapat menerima teks prediksi melalui REST API, WebSocket, atau Server-Sent Events.
- Untuk speech-to-text, aplikasi sudah menyiapkan Web Speech API. Dukungan browser terbaik biasanya ada di Chrome/Edge.

## Menjalankan aplikasi

```bash
npm install
npm run dev
```

Jika PowerShell memblokir `npm.ps1`, gunakan:

```bash
npm.cmd run dev
```

Buka URL yang muncul dari Vite. Untuk uji di HP, gunakan URL network/local IP dari laptop dan pastikan perangkat berada di jaringan yang sama.
