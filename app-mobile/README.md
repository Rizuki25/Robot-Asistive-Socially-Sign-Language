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

Install dependency:

```bash
npm install
```

Jalankan dua terminal.

### Terminal 1: React/Vite

```bash
npm.cmd run dev
```

Vite berjalan di port `5173`. Buka URL network dari laptop di HP, misalnya:

```text
http://192.168.1.10:5173
```

### Terminal 2: Socket.IO Server

```bash
npm.cmd run server
```

Socket.IO berjalan di port `3001` dan dipakai untuk sinkronisasi realtime antar HP/laptop.

Jika PowerShell memblokir `npm.ps1`, gunakan `npm.cmd` seperti contoh di atas.

## Uji realtime banyak HP

1. Pastikan laptop dan semua HP berada di jaringan yang sama, misalnya hotspot HP atau WiFi rumah.
2. Jalankan `npm.cmd run dev` dan `npm.cmd run server` di laptop.
3. Buka app di semua HP menggunakan URL Vite network, misalnya `http://192.168.1.10:5173`.
4. Masuk ke menu `💬 Mode Percakapan`.
5. Pastikan semua perangkat memakai room yang sama, default-nya `demo-ta`.
6. Tekan `🤟 Kirim Hasil` atau `🎤 Balas` dari salah satu perangkat.
7. Pesan akan muncul di semua perangkat yang join room yang sama.

## Endpoint integrasi model robot

Server menyediakan endpoint untuk mengirim hasil model bahasa isyarat ke semua perangkat:

```http
POST http://IP-LAPTOP:3001/api/sign-result
Content-Type: application/json

{
  "roomId": "demo-ta",
  "text": "Saya ingin minum"
}
```

Contoh dengan `curl`:

```bash
curl -X POST http://192.168.1.10:3001/api/sign-result -H "Content-Type: application/json" -d "{\"roomId\":\"demo-ta\",\"text\":\"Saya ingin minum\"}"
```
