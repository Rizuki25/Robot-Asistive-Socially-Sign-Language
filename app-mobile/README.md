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

## Menjalankan dengan Cloudflare Tunnel HTTPS

Gunakan cara ini jika tidak ingin deploy backend ke Render/Railway. Laptop tetap menjalankan frontend dan backend lokal, lalu Cloudflare Tunnel memberi URL HTTPS agar bisa dibuka dari banyak HP.

Kamu membutuhkan 4 terminal aktif:

1. Terminal backend Socket.IO.
2. Terminal tunnel backend.
3. Terminal frontend Vite.
4. Terminal tunnel frontend.

### 1. Jalankan backend Socket.IO

```bash
npm.cmd run server
```

Backend berjalan di:

```text
http://localhost:3001
```

### 2. Buat tunnel untuk backend

Di terminal lain, jalankan `cloudflared`:

```bash
D:\tools\cloudflared.exe tunnel --url http://localhost:3001
```

Salin URL HTTPS yang muncul, misalnya:

```text
https://backend-demo.trycloudflare.com
```

Cek di browser:

```text
https://backend-demo.trycloudflare.com/health
```

Jika berhasil, output-nya:

```json
{"status":"ok"}
```

### 3. Jalankan frontend Vite

```bash
npm.cmd run dev
```

Frontend berjalan di:

```text
http://localhost:5173
```

### 4. Buat tunnel untuk frontend

Di terminal lain:

```bash
D:\tools\cloudflared.exe tunnel --url http://localhost:5173
```

Salin URL HTTPS frontend yang muncul, misalnya:

```text
https://frontend-demo.trycloudflare.com
```

### 5. Buka app di HP

Buka URL frontend dengan tambahan query `socketUrl` berisi URL backend tunnel:

```text
https://frontend-demo.trycloudflare.com/?socketUrl=https%3A%2F%2Fbackend-demo.trycloudflare.com
```

Format umumnya:

```text
URL_FRONTEND_TUNNEL/?socketUrl=URL_BACKEND_TUNNEL_YANG_DI-ENCODE
```

Cara mudah encode URL backend:

- URL backend asli: `https://backend-demo.trycloudflare.com`
- URL backend encoded: `https%3A%2F%2Fbackend-demo.trycloudflare.com`

Setelah pertama kali dibuka, app menyimpan URL backend di browser HP. Selanjutnya kamu bisa membuka URL frontend tunnel biasa selama URL backend tunnel belum berubah.

### 6. Uji banyak HP

1. Buka URL frontend tunnel yang sama di semua HP.
2. Masuk ke `💬 Mode Percakapan`.
3. Pastikan semua perangkat memakai room yang sama, default-nya `demo-ta`.
4. Tekan `🎤 Balas` di salah satu HP.
5. Karena frontend dibuka lewat HTTPS Cloudflare Tunnel, microphone/speech-to-text di HP bisa berjalan.
6. Pesan akan dikirim ke backend tunnel dan muncul di semua perangkat.

Mode percakapan hanya menyimpan tiga pesan terbaru di layar. Ketika pesan baru
masuk, pesan paling lama otomatis dihapus agar halaman tidak perlu di-scroll.

### Catatan penting

- Jangan tutup terminal backend, frontend, dan dua tunnel Cloudflare.
- URL `trycloudflare.com` akan berubah setiap tunnel dijalankan ulang.
- Jika URL backend berubah, buka ulang frontend dengan query `?socketUrl=URL_BACKEND_BARU`.
- Jika muncul halaman warning Cloudflare, klik `Continue`.

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

## Integrasi realtime webcam model ke frontend Vercel

Frontend yang sudah di-deploy tidak perlu menjalankan Vite atau tunnel frontend.
Gunakan alur berikut:

```text
predict_webcam.py -> backend lokal :3001 -> Cloudflare Tunnel -> frontend Vercel
```

Backend menerima:

- `POST /api/sign-result` untuk hasil prediksi yang sudah terkunci.

Hasil prediksi diteruskan lewat Socket.IO hanya ke perangkat yang join room yang
sama. Video kamera tetap diproses dan ditampilkan hanya di laptop; tidak ada
frame yang dikirim ke backend atau HP. Halaman **Bahasa Isyarat -> Suara** hanya
menampilkan hasil teks, tombol membacakan, dan opsi suara otomatis.

### Menjalankan

Terminal backend:

```powershell
cd "D:\Robot-Asistive-Socially-Sign-Language\app-mobile"
npm.cmd run server
```

Terminal tunnel backend:

```powershell
cloudflared tunnel --url http://localhost:3001
```

Buka deployment Vercel dengan URL tunnel backend yang sudah di-encode:

```text
https://robot-asistive-socially-sign-langua.vercel.app/?socketUrl=https%3A%2F%2FURL-RANDOM.trycloudflare.com
```

Terminal model:

```powershell
cd "D:\Robot-Asistive-Socially-Sign-Language\Model"
python -m src.letters.predict_webcam --config configs/letters_90.yaml --model_path outputs/letters_90/models/best_model.pth --motion_threshold 0.001 --pause_frames 15 --min_recording_frames 30 --prediction_interval 3 --vote_window 5 --vote_ratio 0.8 --stable_confidence 0.8 --rearm_motion_frames 3 --server_url http://localhost:3001 --room_id demo-ta --no_speech
```

`--no_speech` menonaktifkan suara lokal dari laptop karena hasil akan dibacakan
oleh browser. Hapus opsi tersebut jika suara lokal tetap dibutuhkan.

### API key opsional

Untuk membatasi endpoint pengiriman model, set nilai yang sama sebelum
menjalankan backend dan model:

```powershell
$env:MODEL_API_KEY="ganti-dengan-kunci-rahasia"
```

Backend tetap dapat dipakai tanpa API key untuk demo lokal.
