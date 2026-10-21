# Suara TTS model di speaker robot AiNex

Status: bridge dan unit test tersedia; **belum diuji dengan speaker fisik**.
Panduan ini memasang jalur audio dari model di laptop ke speaker Logitech X50
yang tersambung ke jack 3,5 mm Raspberry Pi 4B.

## 1. Keputusan jalur audio

Windows tidak melihat speaker robot sebagai perangkat output. X50 tersambung ke
jack audio Pi, bukan ke laptop, jadi ada dua kemungkinan:

| Cara | Prinsip | Keterangan |
|---|---|---|
| **A. Putar di Pi** (dipakai) | Laptop mengirim **nama label**; Pi memutar WAV-nya sendiri | Latensi kecil, tetap jalan tanpa internet, tidak perlu streaming |
| B. Streaming audio | Laptop mengirim sampel audio ke Pi | Perlu server PulseAudio/RTP, lebih rumit, latensi lebih besar |

Cara A dipakai karena 36 WAV sudah tersedia di `Model/assets/letters_audio` dan
`Model/assets/words_audio`. Yang dikirim lewat Wi-Fi hanya satu nama label
(beberapa byte), bukan audio.

Alurnya:

```text
predict_webcam (laptop)
        │  POST /speak {"label": "Halo"}
        ▼
audio_bridge.py (Pi, port 8092)
        │  aplay -D plughw:CARD=Headphones,DEV=0
        ▼
jack 3,5 mm Pi 4B → Logitech X50
```

## 2. Yang perlu disiapkan

- Robot AiNex menyala dan bisa di-SSH (MobaXterm), contoh IP `192.168.50.2`.
- Speaker Logitech X50 dalam kondisi terisi, kabel audio 3,5 mm stereo.
- Folder `/home/ubuntu/ainex_sign_response` sudah ada di robot (hasil langkah
  bridge respons sebelumnya).
- Token yang sama di laptop dan robot, misalnya
  `ainex-demo-halo-baik-2026`. Token ini hanya untuk LAN, jangan diekspos ke
  internet.

## 3. Tahap 1 — Sambungkan speaker dan kenali perangkat ALSA

1. Colok jack 3,5 mm X50 ke **jack audio Pi 4B** (jack bundar di board,
   bukan port HDMI). Gunakan kabel stereo TRS biasa, bukan kabel headset TRRS.
2. Nyalakan X50, putar volumenya ke tengah. Gunakan daya baterainya dahulu;
   mengisi daya sambil memutar dapat menimbulkan dengung.
3. Di MobaXterm (SSH ke robot):

```bash
aplay -l
```

Jika `aplay` atau `speaker-test` belum ada:

```bash
sudo apt install alsa-utils
```

Cari baris seperti:

```text
card 1: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [...]
```

Nomor card bisa `0` atau `1`; yang penting **ID-nya `Headphones`**, karena
perintah di panduan ini memakai `plughw:CARD=Headphones,DEV=0`.

Jika `Headphones` tidak muncul:

```bash
cat /proc/asound/cards
grep -n "dtparam=audio" /boot/firmware/config.txt
```

Pastikan ada `dtparam=audio=on`. Jika belum ada, tambahkan di
`/boot/firmware/config.txt`, lalu reboot:

```bash
sudo nano /boot/firmware/config.txt
sudo reboot
```

## 4. Tahap 2 — Uji bunyi dan atur volume

Masih di SSH robot. Uji nada tanpa menjalankan model:

```bash
speaker-test -D plughw:CARD=Headphones,DEV=0 -c 2 -t sine -f 440 -l 1
```

Harus terdengar nada sekitar 440 Hz bergantian kiri/kanan. Tekan `Ctrl+C`
jika perintah menggantung.

Atur volume kartu Headphones (sesuaikan `-c` dengan nomor card dari `aplay -l`):

```bash
amixer -c 1 sset PCM 90% unmute
```

Jika nama kontrol bukan `PCM`, buka mixer interaktif:

```bash
alsamixer -c 1
```

Naikkan `PCM` dengan tombol panah, lalu `Esc`. Untuk menyimpan volume agar
tetap setelah reboot (opsional):

```bash
sudo alsactl store
```

Catatan volume: keluaran jack Pi kira-kira setingkat line-out. Set `PCM`
sekitar 85–90%, lalu atur keras akhir di knop X50. `PCM` 100% sering membuat
suara pecah dan dengung lebih terdengar.

## 5. Tahap 3 — Salin WAV dan bridge ke robot

Dari **PowerShell laptop, root repository** `D:\Robot-Asistive-Socially-Sign-Language`:

```powershell
scp robot/ainex_sign_response/scripts/audio_bridge.py ubuntu@192.168.50.2:/home/ubuntu/ainex_sign_response/scripts/
scp -r Model/assets/letters_audio Model/assets/words_audio ubuntu@192.168.50.2:/home/ubuntu/ainex_sign_response/audio/
```

Jika folder `ainex_sign_response` belum ada di robot:

```powershell
scp -r robot/ainex_sign_response ubuntu@192.168.50.2:/home/ubuntu/
```

Periksa di robot:

```bash
ls /home/ubuntu/ainex_sign_response/audio/words_audio | wc -l    # harus 10
ls /home/ubuntu/ainex_sign_response/audio/letters_audio | wc -l  # harus 26
ls /home/ubuntu/ainex_sign_response/scripts/audio_bridge.py
```

Bridge hanya memakai pustaka standar Python 3.8, jadi tidak ada `pip install`
di robot.

## 6. Tahap 4 — Jalankan bridge audio

Di MobaXterm, terminal robot:

```bash
export AINEX_RESPONSE_TOKEN='ainex-demo-halo-baik-2026'
python3 /home/ubuntu/ainex_sign_response/scripts/audio_bridge.py
```

Keluaran yang diharapkan:

```text
[AUDIO] 36 label, device plughw:CARD=Headphones,DEV=0, port 8092
```

Biarkan terminal ini terbuka. Bridge berjalan tanpa ROS dan tanpa mengganggu
`response_node.py` di port 8091. Jika robot memakai firewall `ufw`, izinkan
port-nya sekali: `sudo ufw allow 8092/tcp`.

## 7. Tahap 5 — Uji dari laptop

PowerShell laptop:

```powershell
$env:AINEX_RESPONSE_TOKEN='ainex-demo-halo-baik-2026'
$h = @{Authorization = "Bearer $env:AINEX_RESPONSE_TOKEN"}
Invoke-RestMethod -Uri 'http://192.168.50.2:8092/health' -Headers $h
```

Hasil yang diharapkan:

```text
status  device                            labels queued playing
------  ------                            ------ ------ -------
ok      plughw:CARD=Headphones,DEV=0          36      0   False
```

Lalu minta robot mengucapkan satu label:

```powershell
$body = @{label='Halo'} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://192.168.50.2:8092/speak' -Headers $h -ContentType 'application/json' -Body $body
```

Balasan `queued` dan speaker robot harus mengucapkan “Halo”. Coba juga `Baik`,
`A`, dan `Makan`.

Kalau `Invoke-RestMethod` gagal: `401` berarti token berbeda, `404` berarti
label tidak ada di robot, dan `Connection refused` berarti bridge belum jalan
atau IP/port salah.

## 8. Tahap 6 — Hubungkan ke model

Tes dulu hanya audio, tanpa robot bergerak, memakai webcam laptop:

```powershell
cd D:\Robot-Asistive-Socially-Sign-Language\Model
$env:AINEX_RESPONSE_TOKEN='ainex-demo-halo-baik-2026'
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_index 0 --robot_audio_url "http://192.168.50.2:8092"
```

Lakukan isyarat Halo. Audio harus keluar dari speaker robot, bukan speaker
laptop. Opsi `--robot_audio_url` **menggantikan** speaker laptop, jadi tidak
terjadi suara ganda.

Setelah itu, demo lengkap dengan kamera dan gerakan robot:

```powershell
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --robot_url "http://192.168.50.2:8091" --robot_audio_url "http://192.168.50.2:8092"
```

`--no_speech` tidak lagi diperlukan. Jika opsi itu masih ada, speaker laptop
tetap diam dan audio robot tetap dikirim karena `--robot_audio_url` sudah
menggantikan pemutaran lokal.

Opsi ini tersedia di pipeline gabungan `src.combined.predict_webcam` (model 36
kelas yang dipakai demo). Pipeline huruf `src.letters.predict_webcam` belum
diubah.

## 9. Tahap 7 (opsional) — Jalankan bridge otomatis saat boot

Buat file environment token (hanya bisa dibaca user robot):

```bash
printf "AINEX_RESPONSE_TOKEN=ainex-demo-halo-baik-2026\n" > /home/ubuntu/.ainex_audio_env
chmod 600 /home/ubuntu/.ainex_audio_env
```

Buat unit systemd:

```bash
sudo nano /etc/systemd/system/ainex-audio.service
```

Isi:

```ini
[Unit]
Description=AiNex audio bridge (label WAV ke speaker)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
SupplementaryGroups=audio
EnvironmentFile=/home/ubuntu/.ainex_audio_env
ExecStart=/usr/bin/python3 /home/ubuntu/ainex_sign_response/scripts/audio_bridge.py --device plughw:CARD=Headphones,DEV=0
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Aktifkan:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ainex-audio
systemctl status ainex-audio --no-pager
journalctl -u ainex-audio -f
```

Untuk menghentikan sementara: `sudo systemctl stop ainex-audio`.

## 10. Latensi dan sinkronisasi

- POST label lewat LAN biasanya 5–20 ms; `aplay` butuh sekitar 50–150 ms untuk
  mulai berbunyi. Total kira-kira 0,1–0,3 detik setelah hasil terkunci.
- Permintaan audio dan permintaan gerakan dikirim sebagai dua POST terpisah,
  jadi tidak ada jaminan audio berbunyi persis sebelum kepala bergerak.
- Klip diputar berurutan oleh satu worker. Jika hasil datang sangat cepat,
  antrean maksimal 8 klip; klip berikutnya di luar itu ditolak dengan `429`.
- Suara tidak di-streaming, jadi kualitas tidak berubah mengikuti Wi-Fi.
  Wi-Fi yang buruk hanya menambah keterlambatan, bukan memotong audio.

## 11. Pemecahan masalah

| Gejala | Penyebab umum | Tindakan |
|---|---|---|
| `aplay -l` tidak menampilkan `Headphones` | Audio onboard dimatikan | Tambahkan `dtparam=audio=on` di `/boot/firmware/config.txt`, reboot |
| `speaker-test` diam, tidak ada error | Volume 0 / mute, X50 mati, kabel di jack yang salah | `amixer -c 1 sset PCM 90% unmute`; nyalakan X50; pastikan kabel di jack audio Pi |
| Suara hanya di satu sisi | Kabel/plug tidak duduk | Coba kabel lain; pastikan plug masuk penuh; uji ulang dengan `speaker-test -c 2` |
| `Device or resource busy` | PulseAudio/desktop atau proses `aplay` lain memegang kartu | `pactl info`; hentikan pemutar lain; coba `pasuspender -- aplay -D plughw:CARD=Headphones,DEV=0 file.wav` |
| `audio open error: No such file or directory` | ID kartu salah | Jalankan `aplay -L`, cari baris `head`, lalu pakai `plughw:CARD=Headphones,DEV=0` |
| Suara kecil | `PCM` rendah atau X50 volume rendah | Naikkan `PCM` ke 90%, putar knop X50 |
| Suara pecah/berdengung | `PCM` 100%, X50 sedang di-charge, kabel dekat servo | Turunkan `PCM` ke 80–85%; pakai baterai X50; jauhkan kabel audio dari kabel motor |
| `401 unauthorized` | Token laptop ≠ token robot | Samakan `AINEX_RESPONSE_TOKEN` di kedua sisi |
| `Connection refused` | Bridge belum jalan / IP berubah | Cek `systemctl status ainex-audio`, lalu `ss -ltnp` untuk port 8092 |
| Label benar tetapi tidak berbunyi | WAV tidak ada di robot | Cek `/home/ubuntu/ainex_sign_response/audio/words_audio/Halo.wav`; ulangi scp |
| Audio terlambat jauh | Wi-Fi lemah atau Pi sibuk | Ping robot; dekatkan AP; hindari streaming kamera lain |

## 12. Checklist verifikasi

- [ ] `aplay -l` menampilkan `Headphones`.
- [ ] `speaker-test -D plughw:CARD=Headphones,DEV=0 -c 2 ...` terdengar.
- [ ] `amixer` menunjukkan `PCM` tidak mute.
- [ ] `health` dari laptop menjawab `ok` dan `labels` = 36.
- [ ] `POST /speak {"label":"Halo"}` berbunyi di X50.
- [ ] `predict_webcam --camera_index 0 --robot_audio_url ...` berbunyi di robot.
- [ ] Demo penuh (kamera stream + `--robot_url` + `--robot_audio_url`) berjalan.
- [ ] Volume akhir nyaman, kabel aman saat kepala bergerak.

## 13. Kenapa bukan streaming audio

Windows tidak punya sink jaringan seperti PulseAudio. Untuk streaming perlu
perangkat virtual (VB-Cable) atau `ffmpeg`/RTP, lalu server audio di Pi. Itu
menambah 0,3–1 detik latensi, menambah proses yang bisa gagal, dan tetap
membutuhkan laptop menyala. Karena WAV label sudah kecil dan tersedia di
kedua sisi, mengirim nama label lebih sederhana dan lebih tahan gangguan.

Jika nanti ingin suara **selain** WAV label (misalnya musik atau suara browser),
baru pertimbangkan kartu suara USB di Pi atau streaming RTP.

## 14. Berkas yang terlibat

- `robot/ainex_sign_response/scripts/audio_bridge.py` — server audio di Pi.
- `Model/src/common/robot_audio.py` — client non-blocking di laptop.
- `Model/src/combined/predict_webcam.py` — opsi `--robot_audio_url`.
- `robot/tests/test_audio_bridge.py`, `robot/tests/test_robot_audio_client.py`
  — unit test tanpa perangkat audio.
