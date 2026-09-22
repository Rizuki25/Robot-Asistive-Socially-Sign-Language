# Riwayat Percakapan Pengembangan Sistem

Dokumen ini merangkum percakapan mengenai integrasi model bahasa isyarat, pengenalan emosi wajah, aplikasi mobile, Cloudflare Tunnel, audio hasil isyarat, dan respons gerak robot AiNex. Ringkasan disusun berdasarkan urutan pembahasan agar dapat dipakai sebagai catatan kelanjutan proyek.

> Catatan keamanan: nilai token robot yang pernah digunakan tidak ditulis ulang. Gunakan token yang sama pada laptop dan robot melalui variabel `AINEX_RESPONSE_TOKEN`.

## 1. Pemeriksaan pembaruan folder Emotion

Pengguna meminta pemeriksaan perubahan dari teman pada folder `Emotion` dan menanyakan apakah command integrasi isyarat serta emosi masih dapat langsung dijalankan.

Command awal yang dibahas:

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --no_speech --emotion
```

Hasil pemeriksaan:

- Integrasi `--emotion` di folder `Model` memakai `Model/src/common/emotion_recognition.py`.
- Integrasi tersebut tidak menjalankan `Emotion/fusion_webcam.py` secara langsung.
- Folder `Emotion` memiliki model wajah baru dari gabungan FER2013 dan KDEF:
  `Emotion/webcam/models/merged_baseline/weights/best.pt`.
- Versi baru menggunakan input wajah grayscale.
- Label `Netral` tidak boleh menjadi bagian emosi majemuk seperti `Sedih-Netral`.
- Model audio emosi tidak dibutuhkan untuk command gabungan karena integrasi ini hanya memakai pengenalan emosi wajah.

Integrasi di folder `Model` kemudian disesuaikan agar:

- memakai model `merged_baseline` sebagai default;
- mengubah crop wajah menjadi grayscale tiga kanal sebelum inferensi YOLO;
- mencegah `Netral` menjadi label majemuk;
- tetap menjaga tampilan kamera berwarna;
- tetap dapat ditimpa menggunakan argumen `--emotion_model`.

## 2. Menjalankan dua model dengan webcam laptop

Untuk menjalankan pengenalan bahasa isyarat dan emosi wajah menggunakan webcam laptop, command yang digunakan adalah:

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_index 0 --no_speech --emotion
```

Jika kamera yang terbuka salah, `--camera_index 0` dapat diganti menjadi `--camera_index 1`.

## 3. Masalah environment Python dari folder animations

Saat command dijalankan, terminal sedang memakai virtual environment dari folder `animations` dan menghasilkan error:

```text
ModuleNotFoundError: No module named 'cv2'
```

Penyebabnya adalah environment tersebut dibuat untuk Manim dan tidak memiliki dependensi model seperti OpenCV, MediaPipe, PyTorch, dan Ultralytics.

Atas permintaan pengguna:

- folder `animations` beserta `.venv` di dalamnya dihapus;
- aktivasi otomatis environment Python di VS Code dimatikan melalui pengaturan:

```json
{
  "python.terminal.activateEnvironment": false,
  "python-envs.terminal.autoActivationType": "off"
}
```

Setelah perubahan, VS Code perlu menjalankan **Developer: Reload Window** dan membuka terminal baru.

## 4. Percobaan MediaPipe Face Mesh pada model gabungan

Pengguna meminta visualisasi kerangka wajah seperti kerangka tangan. Face Mesh sempat ditambahkan ke program gabungan dan berhasil diuji.

Karena pemrosesan menjadi berat dan FPS turun, fitur Face Mesh kemudian dihapus kembali dari program gabungan. Kondisi akhirnya:

- kerangka tangan tetap ditampilkan;
- pengenalan emosi wajah tetap aktif;
- model emosi baru tetap digunakan;
- kerangka wajah tidak dijalankan pada program gabungan.

Command akhir tetap:

```powershell
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_index 0 --no_speech --emotion
```

## 5. Integrasi app-mobile, Socket.IO, Cloudflare Tunnel, dan audio

Frontend aplikasi sudah tersedia melalui Vercel:

```text
https://robot-asistive-socially-sign-langua.vercel.app/
```

Karena frontend sudah berada di Vercel, Vite lokal dan tunnel frontend tidak diperlukan. Alur sistemnya:

```text
Model laptop
  -> backend Express/Socket.IO lokal pada port 3001
  -> Cloudflare Tunnel backend
  -> frontend Vercel dan perangkat yang bergabung ke room yang sama
```

### Terminal 1 — backend Socket.IO

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\app-mobile
npm.cmd run server
```

Backend dapat diperiksa melalui:

```text
http://localhost:3001/health
```

Respons yang diharapkan:

```json
{"status":"ok"}
```

### Terminal 2 — Cloudflare Tunnel untuk backend

```powershell
cloudflared tunnel --url http://localhost:3001
```

Cloudflare akan memberikan URL acak seperti:

```text
https://NAMA-ACAK.trycloudflare.com
```

Frontend Vercel dibuka dengan URL backend yang sudah di-encode:

```text
https://robot-asistive-socially-sign-langua.vercel.app/?socketUrl=https%3A%2F%2FNAMA-ACAK.trycloudflare.com
```

Room yang digunakan adalah:

```text
demo-ta
```

### Terminal 3 — model dengan webcam laptop

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_index 0 --emotion --web_url http://localhost:3001 --web_room demo-ta
```

Argumen `--no_speech` sengaja tidak dipakai agar file suara hasil isyarat diputar dari speaker laptop.

File audio yang tersedia:

- 26 huruf pada `Model/assets/letters_audio/`;
- 10 kata pada `Model/assets/words_audio/`.

Audio lokal diputar secara asynchronous ketika hasil isyarat terkunci. Aplikasi HP menggunakan text-to-speech browser, bukan file WAV dari laptop.

## 6. Pemeriksaan penurunan FPS

Ketika model isyarat dan emosi dijalankan bersamaan, FPS dapat turun dari sekitar 30 menjadi 18–25 FPS. Pembagian beban program:

- BiLSTM bahasa isyarat menggunakan GPU NVIDIA GeForce RTX 3050 Laptop;
- MediaPipe Hands menggunakan CPU;
- Haar Cascade wajah menggunakan CPU;
- model YOLO emosi secara default menggunakan CPU, kecuali diberikan `--emotion_device 0`;
- pengiriman hasil ke aplikasi berjalan pada thread terpisah;
- audio diputar secara asynchronous.

Pengujian tanpa emosi menunjukkan tampilan kembali lancar. Hal ini menguatkan bahwa pemrosesan emosi menambah beban sistem.

Untuk menguji model emosi pada GPU:

```powershell
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_index 0 --emotion --emotion_device 0 --web_url http://localhost:3001 --web_room demo-ta
```

Untuk mengurangi frekuensi inferensi emosi tanpa mengubah aturan isyarat:

```text
--emotion_interval 0.5
```

Nilai tersebut membatasi pembaruan emosi menjadi sekitar dua kali per detik.

## 7. Analisis isyarat yang kadang tidak langsung ditebak

Video berikut dianalisis:

```text
D:\Robot-Asistive-Socially-Sign-Language\Video Project 3 (1).mp4
```

Temuan dari video:

- durasi video sekitar 9,3 detik dengan sumber 30 FPS;
- tangan sudah terlacak dari sekitar detik pertama;
- status tetap `Siap Isyarat Berikutnya` hingga sekitar detik 6,5;
- perekaman baru terlihat sekitar detik 7;
- pada detik 8 baru terkumpul sekitar 23 dari 45 frame;
- kandidat `Saya` muncul sekitar detik 9 dengan confidence tinggi.

Kesimpulannya, kerangka tangan yang muncul hanya menandakan tangan terlacak. Perekaman baru dimulai jika gerakan melewati `motion_threshold` selama jumlah frame yang disyaratkan. Saat FPS turun, pengumpulan 45 frame juga membutuhkan waktu lebih lama.

Pengguna memilih mempertahankan aturan awal perekaman tanpa perubahan.

## 8. Perbedaan performa saat charger dilepas

Saat charger terpasang, FPS dua model relatif stabil di atas 25. Saat memakai baterai, FPS turun dan muncul delay.

Penjelasan:

- Windows dan firmware laptop membatasi daya CPU/GPU ketika memakai baterai;
- GPU RTX tetap terdeteksi, tetapi clock dan batas dayanya dapat diturunkan;
- MediaPipe dan deteksi wajah juga terdampak karena menggunakan CPU;
- angka FPS pada tampilan adalah kecepatan loop pemrosesan, sehingga ikut turun saat inferensi melambat.

Pengaturan yang disarankan saat menggunakan baterai:

1. Buka **Settings > System > Power & battery**.
2. Ubah **Power mode** menjadi **Best performance**.
3. Nonaktifkan **Energy saver/Battery saver** selama pengujian.
4. Bila masih lambat, gunakan `--emotion_interval 0.5`.

Performa setara charger tidak selalu dapat dijamin karena laptop biasanya menerapkan batas daya perangkat keras saat tidak terhubung ke adaptor.

## 9. Menjalankan program Emotion secara mandiri

Untuk menjalankan pengenalan emosi wajah saja dari webcam laptop, tanpa aplikasi mobile dan tanpa robot:

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Emotion
python fusion_webcam.py --camera 0 --no_speech --no_wave --visual_model webcam/models/merged_baseline/weights/best.pt
```

Arti argumennya:

- `--camera 0`: menggunakan webcam utama laptop;
- `--no_speech`: tidak menjalankan mikrofon dan model emosi audio;
- `--no_wave`: tidak mengirim respons gerak ke robot;
- `--visual_model`: menggunakan model gabungan FER2013 + KDEF.

### Kerangka wajah opsional pada program Emotion

Opsi Face Mesh ditambahkan khusus untuk program mandiri di folder `Emotion`:

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Emotion
python fusion_webcam.py --camera 0 --no_speech --no_wave --visual_model webcam/models/merged_baseline/weights/best.pt --face_mesh
```

Kerangka wajah bersifat visual dan menambah beban komputasi. Hapus `--face_mesh` jika FPS turun. Opsi ini tidak mengaktifkan Face Mesh pada program gabungan di folder `Model`.

## 10. Menghubungkan model dengan kamera robot dan aplikasi mobile

Command gabungan untuk kamera robot serta backend aplikasi:

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --no_speech --emotion --web_url http://localhost:3001 --web_room demo-ta
```

Command tersebut belum mengaktifkan respons gerak robot karena tidak memiliki argumen `--robot_url`.

## 11. Mengaktifkan respons gerak robot AiNex

Server respons pada robot dijalankan melalui terminal SSH/MobaXterm:

```bash
export AINEX_RESPONSE_TOKEN='<TOKEN_YANG_SAMA_DENGAN_LAPTOP>'
python3 /home/ubuntu/ainex_sign_response/scripts/response_node.py _dry_run:=false
```

Log yang menandakan server aktif:

```text
Response bridge port 8091, dry_run=False
```

Pada PowerShell laptop, token yang sama harus diatur kembali karena environment variable di SSH robot tidak otomatis berlaku di laptop:

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
$env:AINEX_RESPONSE_TOKEN='<TOKEN_YANG_SAMA_DENGAN_ROBOT>'
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --no_speech --emotion --web_url http://localhost:3001 --web_room demo-ta --robot_url "http://192.168.50.2:8091"
```

Aturan respons robot:

- hanya label `Halo` dan `Baik` yang memicu gerakan;
- confidence harus minimal 80%;
- `Halo` memanggil gerakan sapaan;
- `Baik` memanggil urutan anggukan;
- label lain seperti `Apa` atau huruf tidak memicu gerakan;
- `--no_speech` hanya mematikan audio keluaran laptop dan tidak mematikan robot.

Jika berhasil, terminal laptop menampilkan pesan seperti:

```text
[ROBOT] Halo: command_sequence_done
```

## 12. Command utama sesuai kebutuhan

### Isyarat dan emosi — webcam laptop, tanpa app

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_index 0 --no_speech --emotion
```

### Isyarat, emosi, audio lokal, dan app-mobile

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_index 0 --emotion --web_url http://localhost:3001 --web_room demo-ta
```

### Emosi wajah saja

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Emotion
python fusion_webcam.py --camera 0 --no_speech --no_wave --visual_model webcam/models/merged_baseline/weights/best.pt
```

### Isyarat, emosi, app-mobile, dan respons robot

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
$env:AINEX_RESPONSE_TOKEN='<TOKEN_YANG_SAMA_DENGAN_ROBOT>'
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --no_speech --emotion --web_url http://localhost:3001 --web_room demo-ta --robot_url "http://192.168.50.2:8091"
```

## 13. Catatan operasional

- Jalankan hanya satu program yang memakai webcam pada satu waktu.
- Jangan menyalin URL dari Markdown dalam bentuk `[teks](tautan)` ke PowerShell. Gunakan URL polos di dalam tanda kutip.
- Quick Tunnel Cloudflare memperoleh URL baru setiap dijalankan ulang; perbarui parameter `socketUrl` pada frontend Vercel.
- Gunakan room `demo-ta` secara konsisten pada model dan aplikasi.
- Jangan menjalankan ulang request gerak robot secara manual jika statusnya tidak pasti; periksa terminal robot terlebih dahulu untuk menghindari duplikasi gerakan.
- Untuk keluar dari program kamera, tekan `Q` atau `Esc`.

