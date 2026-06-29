# AGENTS.md — App Mobile Context

Instruksi dan konteks ini berlaku untuk folder `app-mobile`.

## Ringkasan project

`app-mobile` adalah aplikasi mobile web berbasis React + Vite + Tailwind untuk tugas akhir pengenalan bahasa isyarat.

Nama aplikasi:

```text
Sign Language Assistant
```

Tujuan aplikasi:

- Menjadi interface mobile untuk komunikasi dua arah antara penyandang disabilitas dan orang normal.
- Menerima hasil pengenalan bahasa isyarat dari robot/server model.
- Menampilkan hasil sebagai teks dan membacakan teks dengan text-to-speech.
- Mengubah suara orang normal menjadi teks dengan speech-to-text agar dapat dibaca penyandang disabilitas.
- Mendukung realtime multi-device menggunakan Socket.IO.

## Mode aplikasi

Aplikasi memiliki 3 menu utama:

1. `🤟 Bahasa Isyarat → Suara`
   - Input berasal dari robot/server model pengenalan bahasa isyarat.
   - Output berupa teks.
   - Teks dapat dibacakan dengan text-to-speech.
   - Ada tombol untuk mengirim hasil ke semua perangkat melalui Socket.IO.

2. `🎤 Suara → Teks`
   - Input suara orang normal dari mikrofon browser.
   - Output berupa teks.
   - Hasil dikirim realtime ke semua perangkat yang join room sama.

3. `💬 Mode Percakapan`
   - Mode utama untuk demo komunikasi dua arah.
   - Menampilkan chat realtime:
     - bubble kiri: hasil bahasa isyarat dari penyandang disabilitas/robot/model
     - bubble kanan: hasil speech-to-text dari orang normal
   - Semua HP/laptop yang join room sama melihat pesan yang sama.

## Arsitektur saat ini

Frontend:

```text
React + Vite + Tailwind
Port lokal: 5173
```

Backend realtime:

```text
Express + Socket.IO
File: server/index.js
Port lokal: 3001
```

Room default:

```text
demo-ta
```

Event Socket.IO utama:

- `join-room`
- `send-message`
- `message`

Format message:

```js
{
  roomId: 'demo-ta',
  sender: 'sign' | 'voice' | 'system',
  text: 'Isi pesan',
  timestamp: Date.now()
}
```

## Keputusan penting

### Tidak menggunakan Render/Railway untuk backend

Sempat dipertimbangkan deploy backend Socket.IO ke Render/Railway, tetapi user batal karena platform meminta kartu pembayaran walaupun memilih free plan.

Keputusan akhir:

- Tidak menggunakan Render/Railway.
- Menggunakan Cloudflare Tunnel untuk memberi HTTPS pada frontend dan backend lokal.
- Laptop tetap menjalankan frontend dan backend lokal.
- Banyak HP mengakses aplikasi lewat URL Cloudflare Tunnel.

### Mengapa perlu HTTPS?

Speech-to-text/microphone di browser HP biasanya membutuhkan secure context.

- Di laptop, `http://localhost:5173` dianggap aman.
- Di HP, `http://IP-LAPTOP:5173` tidak dianggap aman.
- Karena itu tombol `🎤 Balas` di HP bisa langsung berhenti jika app dibuka lewat HTTP lokal.
- Solusi saat ini: buka frontend melalui HTTPS Cloudflare Tunnel.

### Socket.IO URL

Frontend membaca URL backend Socket.IO dari beberapa sumber:

1. Query string:

```text
?socketUrl=https%3A%2F%2Fbackend-demo.trycloudflare.com
```

2. Environment variable:

```text
VITE_SOCKET_URL
```

3. Local network default:

```text
http://hostname:3001
```

Jika `socketUrl` diberikan lewat query, app menyimpannya ke `localStorage` dengan key:

```text
sign-language-socket-url
```

Jadi setelah pertama kali dibuka, user tidak perlu menambahkan query lagi selama URL backend tunnel belum berubah.

## Cara menjalankan lokal biasa

Gunakan 2 terminal.

Terminal 1 — frontend:

```powershell
cd "D:\Kuliah\TUGAS AKHIR\SignLanguage_BiLSTM\app-mobile"
npm.cmd run dev
```

Terminal 2 — backend:

```powershell
cd "D:\Kuliah\TUGAS AKHIR\SignLanguage_BiLSTM\app-mobile"
npm.cmd run server
```

Frontend:

```text
http://localhost:5173
```

Backend:

```text
http://localhost:3001
```

Lokal biasa cocok untuk laptop, tetapi speech-to-text di HP kemungkinan gagal jika dibuka lewat `http://IP-LAPTOP:5173`.

## Cara menjalankan dengan Cloudflare Tunnel

Gunakan 4 terminal aktif.

### Terminal 1 — backend Socket.IO

```powershell
cd "D:\Kuliah\TUGAS AKHIR\SignLanguage_BiLSTM\app-mobile"
npm.cmd run server
```

Backend lokal:

```text
http://localhost:3001
```

### Terminal 2 — tunnel backend

```powershell
cd D:\tools
.\cloudflared.exe tunnel --url http://localhost:3001
```

Contoh output:

```text
https://backend-demo.trycloudflare.com
```

Health check:

```text
https://backend-demo.trycloudflare.com/health
```

Expected response:

```json
{"status":"ok"}
```

### Terminal 3 — frontend Vite

```powershell
cd "D:\Kuliah\TUGAS AKHIR\SignLanguage_BiLSTM\app-mobile"
npm.cmd run dev
```

Frontend lokal:

```text
http://localhost:5173
```

### Terminal 4 — tunnel frontend

```powershell
cd D:\tools
.\cloudflared.exe tunnel --url http://localhost:5173
```

Contoh output:

```text
https://frontend-demo.trycloudflare.com
```

### URL yang dibuka di HP

Buka frontend tunnel dengan query `socketUrl` berisi backend tunnel yang sudah di-encode:

```text
https://frontend-demo.trycloudflare.com/?socketUrl=https%3A%2F%2Fbackend-demo.trycloudflare.com
```

Format umum:

```text
URL_FRONTEND_TUNNEL/?socketUrl=URL_BACKEND_TUNNEL_ENCODED
```

Contoh encoding:

```text
https://backend-demo.trycloudflare.com
```

menjadi:

```text
https%3A%2F%2Fbackend-demo.trycloudflare.com
```

Catatan:

- Jangan tutup terminal backend, frontend, dan kedua tunnel.
- URL `trycloudflare.com` berubah setiap tunnel dijalankan ulang.
- Jika URL backend berubah, buka ulang frontend dengan query `?socketUrl=URL_BACKEND_BARU`.
- Jika muncul warning Cloudflare, klik `Continue`.

## Endpoint integrasi robot/model

Backend menyediakan endpoint untuk server robot/model mengirim hasil prediksi bahasa isyarat:

```http
POST http://localhost:3001/api/sign-result
Content-Type: application/json

{
  "roomId": "demo-ta",
  "text": "Saya ingin minum"
}
```

Jika menggunakan tunnel backend, endpoint menjadi:

```http
POST https://BACKEND_TUNNEL.trycloudflare.com/api/sign-result
Content-Type: application/json

{
  "roomId": "demo-ta",
  "text": "Saya ingin minum"
}
```

Saat endpoint dipanggil, pesan akan dibroadcast ke semua perangkat yang join room sama.

## File penting

- `src/App.jsx`
  - UI utama, mode aplikasi, speech-to-text, text-to-speech, Socket.IO client.

- `server/index.js`
  - Express + Socket.IO server.
  - Endpoint `/health`.
  - Endpoint `/api/sign-result`.

- `README.md`
  - Dokumentasi penggunaan umum.

- `.env.example`
  - Contoh environment variable.

- `vite.config.js`
  - Konfigurasi Vite.
  - `allowedHosts: true` sudah ditambahkan agar Cloudflare Tunnel tidak diblokir.

## Validasi terakhir yang pernah berhasil

Frontend build:

```bash
npm run build --prefix app-mobile
```

Server start:

```bash
npm run server --prefix app-mobile
```

Output server yang benar:

```text
Socket.IO server running on http://0.0.0.0:3001
```

## Preferensi implementasi berikutnya

Jika melanjutkan pengembangan:

- Pertahankan UI sederhana dan mobile-first.
- Jangan hapus mode yang sudah ada.
- Prioritaskan `💬 Mode Percakapan` sebagai mode demo utama.
- Integrasi robot/model sebaiknya masuk melalui endpoint `/api/sign-result` dulu.
- Jika menambah fitur realtime, gunakan room `demo-ta` sebagai default agar demo mudah.
- Hindari menambahkan dependency besar jika tidak perlu.
