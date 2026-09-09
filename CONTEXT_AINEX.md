# Ringkasan kelanjutan proyek AiNex

Diperbarui: 9 September 2026. Ini ringkasan keputusan dan bukti pengujian,
bukan transkrip. Baca juga `robot/README.md` dan kode aktual sebelum mengubahnya.

## Tujuan dan cara bekerja dengan pengguna

- Robot pameran: pengenalan bahasa isyarat, teks di aplikasi mobile, suara robot,
  serta respons sosial. Pengenalan ekspresi wajah akan ditambahkan oleh teman pengguna.
- Berikan langkah **satu tahap dahulu**, tunggu hasil sebelum tahap selanjutnya.
- Pengguna memakai **MobaXterm untuk SSH robot** dan **PowerShell/VS Code di laptop**.
  Selalu sebutkan lokasi menjalankan perintah. `scp` tidak otomatis membuka sesi SSH.
- Pengguna menjalankan `python` langsung di laptop, bukan Conda.
- Ada perubahan pengguna pada worktree; jangan reset/menimpa perubahan tak terkait.

## Perangkat dan koneksi yang sudah diperiksa

- AiNex Raspberry Pi **4B**, lingkungan aktual **Ubuntu 20.04.6, aarch64,
  Python 3.8.10, ROS Noetic**. Jangan mengasumsikan Debian/Docker berdasarkan panduan Pi 5.
- Robot terakhir: `192.168.50.2`, akun `ubuntu`; ROS hostname `ubuntu`.
  IP bisa berubah. Workspace robot: `/home/ubuntu/ros_ws`.
- MediaPipe robot 0.10.5 (`solutions` tersedia), OpenCV 4.7.0, CvBridge tersedia.
- Kamera USB: alias `/dev/usb_cam`; node `/camera`; gambar `/camera/image_raw`,
  640×480 `rgb8`, sekitar 22.59 FPS. Versi compressed JPEG quality 80 sekitar 1.2 MB/s.
- Server web port **8080 sering sudah aktif dari bawaan**. `Address already in use`
  bukan alasan menghentikan sembarang proses. Jangan membuat driver kamera kedua.
- ALSA mendeteksi HDMI dan `Headphones`; speaker fisik belum dikonfirmasi berbunyi.

## Arsitektur yang dipilih dan hasil uji

**Kamera robot → HTTP compressed stream → MediaPipe + model di laptop → respons ROS.**
Awalnya ingin mengirim landmark dari Pi, tetapi benchmark Pi hanya 3–5 FPS (model
MediaPipe penuh), 5–9 FPS (ringan); resize 320×240 tidak banyak membantu.
Laptop RTX 3050 diuji sekitar 30 FPS dengan webcam lokal. Stream robot + klasifikasi
laptop juga telah dicoba pengguna dan dinyatakan cukup lancar serta menebak dengan baik.
Belum ada pengukuran formal latensi kamera-ke-output.

- Model: `Model/outputs/combined_90/models/best_model.pth`;
  konfigurasi `Model/configs/combined_90.yaml`, encoder di
  `Model/dataset/combined/processed_90/label_encoder.json`.
- 36 kelas: 26 huruf + 10 kata, termasuk Halo dan Baik. Landmark mentah `(42,3)`,
  slot Left lalu Right; preprocessing menjadi 140 fitur, maksimal 90 frame.
- `predict_webcam.py` masih menunggu minimal **45 frame** sebelum inferensi berkala;
  jangan menyamakan FPS tampilan dengan waktu sampai hasil terkunci.
- Fitur memakai posisi/perpindahan wrist global: tracking kepala saat pengenalan
  dapat mengganggu. Pengguna memilih sudut kepala tetap, tanpa tracking.
- Mobile punya endpoint `POST /api/sign-result` pada port 3001 dan Socket.IO room
  `demo-ta`; integrasi hasil model ke mobile belum diverifikasi. Baca `app-mobile/AGENTS.md`.

## Posisi kepala dan respons gerakan

- Pengguna sudah memilih **position 0.30** sebagai sudut kamera yang cocok.
- Topic `/head_tilt_controller/command`, tipe `ainex_interfaces/HeadState`:
  `float64 position`, `float64 duration`. Duration dalam **detik**; posisi dikonversi
  `angle2pulse()` oleh controller. Jangan mengganti 0.30 dengan tick servo 500.
- `Baik`: dua anggukan **0.30 → 0.20 → 0.40 → 0.20 → 0.40 → 0.30**,
  0.8 detik per perpindahan. **Sudah berhasil bergerak melalui POST dari laptop**.
- `Halo`: dry-run POST berhasil memilih `greet`, tetapi gerakan fisik belum diuji.
  Tersedia `greet.d6a` dan `wave.d6a` di
  `/home/ubuntu/software/ainex_controller/ActionGroups`.
  Greet mempunyai 8 frame dan kolom servo 1–22, termasuk kaki, tanpa kepala 23–24.
  Jangan menganggap greet hanya menggerakkan lengan atau mengaktifkan Halo tanpa tes.
- API Pi 4B telah diperiksa: `MotionManager(action_path=...)`,
  `set_servos_position(duration_ms, *args)`, `run_action(name)` blocking.
  Constructor membuka serial; mode Baik saja kini menggunakan publisher ROS kepala,
  tidak membuat MotionManager terpisah.
- Config sekarang: `hardware_verified: true`, **`live_labels: [Baik]`**;
  default startup tetap dry-run, live hanya dengan `_dry_run:=false`.
- Respons hanya hasil terkunci confidence >=0.8; ID unik, tanpa retry gerakan,
  penolakan saat sibuk, jeda. Laptop menahan klasifikasi tetapi membaca video
  selama respons. Ctrl+C laptop tidak menghentikan gerakan yang sudah diterima robot.

## File implementasi dan status tes

- `Model/src/combined/predict_webcam.py`: opsi `--camera_url`, `--robot_url`,
  log `[STARTUP]`, FFmpeg timeout open 10 detik / read 5 detik.
- `Model/src/common/robot_response.py`: client HTTP async + heartbeat readiness.
- `robot/ainex_sign_response/`: package ROS, config, launch, `response_node.py`,
  `inspect_robot.py` (read-only), `ready_audio_node.py`.
- Bridge disalin ke **`/home/ubuntu/ainex_sign_response`**, dijalankan langsung dengan
  Python dalam terminal ROS; **belum diinstal ke catkin/autostart**. Port **8091**.
- Tes software: `python -m unittest discover -s robot/tests -v` dari root repo.
  Tes disiapkan, tetapi **belum berhasil dieksekusi oleh agent**: terminal tool tidak
  menemukan interpreter Python (launcher py ada, tidak menemukan instalasi).
  Diff/XML diperiksa; jangan mengklaim semua unit test sudah lulus.

## Pengumuman suara kesiapan

- Kalimat: **“Halo, robot pengenalan bahasa isyarat dan ekspresi wajah sudah siap digunakan.”**
- Voice Edge TTS `id-ID-GadisNeural`. Generator:
  `python -m src.common.generate_ready_audio` dari `Model`;
  dependency generation `edge-tts imageio-ffmpeg`, internet diperlukan saat membuat.
- Target `robot/ainex_sign_response/audio/ready.wav`; saat ringkasan dibuat **belum ada
  WAV di workspace** (hanya README). Pembuatan, penyalinan, dan speaker belum diuji.
- Playback offline via `aplay`, candidate device `plughw:CARD=Headphones,DEV=0`.
- Node audio menunggu frame kamera + heartbeat sign + heartbeat expression segar
  (<6 detik), sekali per masa hidup node. Sign heartbeat sudah di client;
  **program ekspresi belum tersedia/terhubung**, jangan memalsukan readiness.
- Suara hasil klasifikasi di speaker robot juga belum diimplementasikan;
  `--no_speech` mengatur audio hasil di laptop, bukan audio kesiapan robot.

## Command yang biasa dipakai

PowerShell laptop, root repo (sesuaikan IP):

```powershell
scp -r robot/ainex_sign_response ubuntu@192.168.50.2:/home/ubuntu/
```

MobaXterm, robot, posisi kamera:

```bash
rostopic pub -1 /head_tilt_controller/command ainex_interfaces/HeadState "{position: 0.30, duration: 1.0}"
```

Server kamera jika belum aktif, terminal robot tersendiri:

```bash
rosrun web_video_server web_video_server
```

Bridge respons kepala, terminal robot tersendiri:

```bash
export AINEX_RESPONSE_TOKEN='ainex-demo-halo-baik-2026'
python3 /home/ubuntu/ainex_sign_response/scripts/response_node.py _dry_run:=false
```

Token di atas contoh demo LAN yang sudah digunakan, bukan kredensial produksi.
Gunakan token sama di laptop; jangan expose bridge ke internet.

PowerShell laptop:

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
$env:AINEX_RESPONSE_TOKEN='ainex-demo-halo-baik-2026'
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --robot_url "http://192.168.50.2:8091" --no_speech
```

Browser test: `http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed`.
Tutup tab stream saat menjalankan model. Jangan paste URL dalam format `[teks](url)` ke CLI.

## Kondisi terakhir dan langkah berikutnya

Percobaan model + respons otomatis tertahan setelah pesan GPU; browser juga kosong.
Diagnosis: `/camera/image_raw` **no new messages**, ping `/camera` connection refused;
hanya `/dev/video10–16` internal codec/ISP terdeteksi, tidak ada kamera USB.
**Pengguna akhirnya menyadari USB kamera BELUM DICOLOKKAN.** Jangan menuduh model,
servo, Wi-Fi, atau kualitas kamera sebagai akar masalah ini.

Instruksi terakhir: USB eksternal kamera boleh dipasang saat robot menyala dan diam,
kabel cukup longgar untuk kepala. Tunggu sekitar 5 detik, lalu di MobaXterm:

```bash
ls -l /dev/usb_cam /dev/video*
rostopic hz /camera/image_raw
```

**Hasil setelah USB dipasang belum diterima.** Lanjut dari sini: pastikan perangkat
dan frame kembali; jika node tidak pulih, periksa/restart node kamera yang tepat,
jangan menjalankan driver kedua atau mengganti device ke video10. Setelah stream
kembali, ulangi pengenalan **Baik → anggukan** (belum terkonfirmasi end-to-end).
Berikutnya uji/aktifkan Halo, pengumuman kesiapan, dan integrasi mobile/ekspresi.
