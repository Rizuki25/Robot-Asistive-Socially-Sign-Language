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

## Deploy online HTTPS

> Catatan penting: Vercel cocok untuk deploy frontend React/Vite, tetapi tidak cocok untuk Socket.IO server yang harus berjalan terus. Untuk realtime multi-HP, deploy frontend ke Vercel dan deploy `server/index.js` ke layanan backend yang mendukung WebSocket seperti Render atau Railway.

### 1. Deploy Socket.IO backend

Deploy folder `app-mobile` ke Render/Railway sebagai Web Service.

Pengaturan umum:

```text
Root Directory : app-mobile
Build Command  : npm install
Start Command  : npm run server
Port           : 3001 atau otomatis dari environment PORT
```

Setelah deploy, kamu akan mendapatkan URL backend HTTPS, misalnya:

```text
https://sign-language-socket.onrender.com
```

Cek health endpoint:

```text
https://sign-language-socket.onrender.com/health
```

Jika berhasil, output-nya:

```json
{"status":"ok"}
```

### 2. Deploy frontend ke Vercel

Di Vercel:

```text
Framework Preset : Vite
Root Directory   : app-mobile
Build Command    : npm run build
Output Directory : dist
```

Tambahkan Environment Variable di Vercel:

```text
VITE_SOCKET_URL=https://URL-BACKEND-SOCKET-KAMU
```

Contoh:

```text
VITE_SOCKET_URL=https://sign-language-socket.onrender.com
```

Setelah itu klik deploy. Frontend akan mendapatkan URL HTTPS seperti:

```text
https://sign-language-assistant.vercel.app
```

### 3. Atur CORS backend

Setelah frontend Vercel jadi, sebaiknya tambahkan environment variable di backend:

```text
CLIENT_ORIGIN=https://URL-FRONTEND-VERCEL-KAMU
```

Contoh:

```text
CLIENT_ORIGIN=https://sign-language-assistant.vercel.app
```

Lalu restart/redeploy backend.

### 4. Uji di HP

1. Buka URL Vercel di beberapa HP.
2. Masuk ke `💬 Mode Percakapan`.
3. Pastikan semua HP memakai room yang sama, misalnya `demo-ta`.
4. Tekan `🎤 Balas` di salah satu HP.
5. Karena URL sudah HTTPS, fitur microphone/speech-to-text lebih aman untuk browser HP.
6. Hasil teks akan dikirim ke Socket.IO backend dan muncul di semua perangkat.

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
