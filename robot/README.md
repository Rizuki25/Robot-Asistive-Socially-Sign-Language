# Respons sosial AiNex: Halo dan Baik

Status: bridge laptop/ROS dan urutan anggukan telah dibuat, tetapi API serta
gerakan fisik pada AiNex 4B pengguna belum diverifikasi. Default **dry-run**:
tidak mengimpor driver servo dan tidak menggerakkan robot.

Alur: hasil terkunci `predict_webcam.py` → HTTP port 8091 → node ROS
`sign_response` → `MotionManager`. Video tetap melalui port 8080.
Tidak perlu ROS pada laptop. Tidak ada perubahan checkpoint atau training.

- Halo: memanggil action bawaan `greet.d6a`. Keberadaan dan isi gerakan harus
  dicek pada robot; tidak ada file gerakan baru yang menimpa action bawaan.
- Baik: urutan center → center-amplitude → center+amplitude, dua kali,
  lalu center. Kandidat default: servo tilt 24, center 500, amplitude 35,
  500 ms per perpindahan. Ini satuan posisi servo, bukan derajat.
  Nilai ini belum dikalibrasi pada robot pengguna.
- Hanya label Halo/Baik dengan confidence >= 0.8; gerakan diserialkan,
  request saat sibuk ditolak, ID duplikat diabaikan (256 ID terakhir per proses).
- Laptop menahan klasifikasi saat request berlangsung dan dua detik sesudahnya,
  sambil tetap membaca video. Jika koneksi gagal: jeda 30 detik dan tidak retry.
  Jangan restart/retry manual sebelum memastikan robot sudah selesai.
- Status `command_sequence_done` berarti API/urutan perintah selesai;
  bukan verifikasi posisi fisik atau pengukuran latensi kamera-ke-gerakan.

## 1. Salin ke robot dan periksa tanpa gerakan

PowerShell laptop, dari root repository (sesuaikan IP jika berubah):

```powershell
scp -r robot/ainex_sign_response ubuntu@192.168.50.2:/home/ubuntu/
```

SSH robot:

```bash
python3 /home/ubuntu/ainex_sign_response/scripts/inspect_robot.py
```

Skrip hanya membaca source MotionManager, daftar action, database greet secara
read-only, dan referensi konfigurasi kepala. Kirim output untuk memastikan API
versi 4B, durasi dalam ms, run_action blocking, action greet yang benar,
dan posisi kepala yang sesuai. Jangan menjalankan contoh demo kicking/walking
untuk memeriksa API.

## 2. Uji koneksi dalam dry-run

Gunakan token acak yang sama pada robot dan laptop. Minimal 16 karakter.
Jangan expose port ini lewat Cloudflare Tunnel/internet; ini bridge LAN.

SSH robot (gunakan terminal yang ROS-nya sudah aktif):

```bash
export AINEX_RESPONSE_TOKEN='GANTI_DENGAN_TOKEN_ACAK_YANG_SAMA'
python3 /home/ubuntu/ainex_sign_response/scripts/response_node.py
```

Tidak perlu memasang package ke catkin untuk menjalankan skrip langsung;
source workspace AiNex harus tersedia sebagaimana terminal ROS sebelumnya.

PowerShell laptop:

```powershell
$env:AINEX_RESPONSE_TOKEN='GANTI_DENGAN_TOKEN_ACAK_YANG_SAMA'
cd D:\Robot-Asistive-Socially-Sign-Language\Model
python -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --robot_url "http://192.168.50.2:8091" --no_speech
```

Server kamera 8080 harus tetap aktif. Setelah isyarat Halo/Baik terkunci,
terminal robot melaporkan `started` dan `dry_run_done`; robot tidak bergerak.
Fallback confidence rendah tetap boleh tampil sebagai hasil model tetapi tidak
memicu robot. Model biasa tanpa `--robot_url` tidak mengirim perintah.

Status ROS (terminal robot lain):

```bash
rostopic echo /sign_response/status
```

## 3. Aktifkan gerakan setelah verifikasi perangkat

Pastikan `greet` benar-benar gerakan yang dikehendaki dan postur awal/akhirnya
sesuai. Verifikasi center/rentang tilt dan API robot sebelum mengubah
`hardware_verified: true` di config/responses.yaml. Jika greet tidak ada atau
API berbeda, adaptasi diperlukan; jangan menggantinya dengan action acak.

Uji satu gerakan saat robot ditopang pada posisi berdiri yang sesuai, hentikan
mode berjalan serta face tracking/autonomous demo yang dapat memberi perintah
servo bersamaan. Bridge ini tidak mengunci pengendali ROS lain secara global.
Pertahankan layanan driver dan kamera yang diperlukan.

Setelah verifikasi, restart node dengan:

```bash
python3 /home/ubuntu/ainex_sign_response/scripts/response_node.py _dry_run:=false
```

Untuk tes gerakan tanpa model, PowerShell laptop:

```powershell
$body = @{event_id=[guid]::NewGuid().ToString(); label='Halo'; confidence=1.0} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://192.168.50.2:8091/respond' -Headers @{Authorization="Bearer $env:AINEX_RESPONSE_TOKEN"} -ContentType 'application/json' -Body $body
```

Tes `Baik` terpisah dengan event_id baru setelah gerakan selesai. Gerakan nod
mengembalikan kepala ke center yang dikonfigurasi, bukan membaca posisi semula.
Jangan menjalankan dua node bridge sekaligus. Ctrl+C pada laptop **bukan**
emergency stop: action yang sudah diterima robot dapat terus berjalan. Ctrl+C
pada node juga tidak dijamin membatalkan run_action bawaan yang sedang aktif.
Gunakan kendali berhenti/daya robot bila gerakan perlu dihentikan segera.

Opsional untuk penggunaan catkin kemudian: salin package ke workspace khusus,
build dan source, lalu `roslaunch ainex_sign_response responses.launch`.
Default launch tetap dry-run; tidak ada instalasi atau autostart otomatis.

## Pemeriksaan software tanpa robot

```powershell
python -m unittest discover -s robot/tests -v
python -m py_compile Model/src/common/robot_response.py Model/src/combined/predict_webcam.py robot/ainex_sign_response/scripts/response_node.py robot/ainex_sign_response/scripts/inspect_robot.py
```

Tests menggunakan stub ROS dan tidak menguji mekanik, kamera, jaringan robot,
atau kompatibilitas MotionManager terpasang. Suara hasil klasifikasi dan aplikasi
mobile belum termasuk perubahan ini; pengumuman kesiapan dijelaskan di bawah.

## Pengumuman kesiapan dengan suara Gadis

Kalimat: **Halo, robot pengenalan bahasa isyarat dan ekspresi wajah sudah siap digunakan.**
Suara `id-ID-GadisNeural`, dibuat sekali menggunakan Edge TTS di laptop.
Robot memutar WAV lokal sehingga tidak membutuhkan internet saat pameran.

PowerShell laptop, dalam folder Model:

```powershell
python -m pip install edge-tts imageio-ffmpeg
python -m src.common.generate_ready_audio
cd ..
scp -r robot/ainex_sign_response ubuntu@192.168.50.2:/home/ubuntu/
```

Generator menolak menimpa file yang sudah ada. Jalankan hanya sekali; jika
ready.wav sudah tersedia, cukup salin ke robot. Tidak perlu mengubah audio label.

Uji speaker secara terpisah di SSH robot:

```bash
aplay -D plughw:CARD=Headphones,DEV=0 /home/ubuntu/ainex_sign_response/audio/ready.wav
```

Output Headphones sesuai perangkat ALSA yang sebelumnya terdeteksi. Ini memerlukan
speaker aktif yang tersambung ke jack audio; daftar aplay belum membuktikan speaker
terhubung. Untuk USB/HDMI, pilih perangkat aktual dari `aplay -l`.

Setelah tes audio berhasil, jalankan node di SSH lain:

```bash
python3 /home/ubuntu/ainex_sign_response/scripts/ready_audio_node.py
```

Untuk catkin: `roslaunch ainex_sign_response responses.launch ready_audio:=true`.
Jangan jalankan node langsung dan launch bersamaan. Default servo tetap dry-run;
audio kesiapan tetap dapat berbunyi meskipun servo dry-run.

Pengumuman hanya sekali per masa hidup node, setelah tiga sinyal segar (<6 detik):
- Frame pada `/camera/image_raw` diterima.
- `/sign_language/ready` bernilai true. Program laptop dengan `--robot_url`
  otomatis mengirim heartbeat setelah frame berhasil diproses MediaPipe,
  sementara model sudah dimuat. Ini kesiapan pipeline, bukan ukuran akurasi.
- `/expression/ready` bernilai true dari program teman Anda. Belum terhubung
  otomatis karena source program ekspresi belum tersedia.

Program ekspresi dapat menerbitkan Bool ROS pada `/expression/ready` setiap
2 detik saat kamera, model ekspresi, dan pemrosesannya aktif. Jika berjalan
di laptop, panggil endpoint bridge dengan token yang sama:

```python
import json
import os
from urllib.request import Request, urlopen

def expression_heartbeat(ready=True):
    request = Request(
        'http://192.168.50.2:8091/readiness',
        data=json.dumps({'component': 'expression', 'ready': ready}).encode('utf-8'),
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + os.environ['AINEX_RESPONSE_TOKEN']},
        method='POST')
    with urlopen(request, timeout=2) as response:
        response.read()
```

Panggil dari worker jaringan setiap 2 detik setelah pemrosesan frame berhasil,
bukan sekadar saat proses dimulai; tangani error koneksi agar loop kamera tidak
berhenti. Kirim false saat berhenti/error, dan hentikan heartbeat jika pemrosesan
macet. Sinyal yang tidak diperbarui akan kedaluwarsa. Jangan memalsukan readiness
ekspresi hanya untuk memicu suara. Untuk mendengarkan audio saja, gunakan aplay.

Node audio tidak otomatis menyatakan ekspresi siap ketika hanya kamera aktif.
Jika playback gagal, lihat log dan periksa perangkat/file sebelum restart node.
Restart node mengizinkan satu pengumuman baru. Playback ini terpisah dari
respons servo; belum ada pengatur prioritas bersama untuk suara lain.

Referensi vendor (perlu cocokkan dengan versi 4B yang terpasang):
- https://wiki.hiwonder.com/projects/AiNex/en/raspberry-pi5-version/docs/5.ROS%20Robot%20Control%20Course.html
- https://wiki.hiwonder.com/projects/AiNex/en/raspberry-pi5-version/docs/6.ROS%20Robot%20AI%20Vision%20Course.html
