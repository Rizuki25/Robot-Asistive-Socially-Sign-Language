# 🤟 Sign Language Recognition using BiLSTM

Sistem pengenalan bahasa isyarat menggunakan **Bidirectional LSTM (BiLSTM)** dengan **PyTorch**. Pipeline huruf mendukung gestur BISINDO satu maupun dua tangan sesuai `configs/letters.yaml`.

## 📋 Deskripsi

Project ini menggunakan **MediaPipe Hands** untuk mengekstrak 21 titik landmark per tangan dari video bahasa isyarat, kemudian melatih model **BiLSTM** untuk mengklasifikasikan gestur tangan. Model yang sudah dilatih juga dapat digunakan untuk memprediksi satu video dan menampilkan hasil prediksi kelas huruf langsung pada video.

## 🏗️ Arsitektur Pipeline

```
Video Dataset → Ekstraksi Landmark → Preprocessing → Training BiLSTM → Evaluasi → Prediksi Video
                  (MediaPipe)         (Normalisasi)    (PyTorch)       (Metrics)   (OpenCV)
```

## 🚀 Quick Start

> Semua perintah di bawah dijalankan dari dalam folder `Model/` (`cd Model`).

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

### 2. Siapkan Dataset Huruf

Tempatkan video bahasa isyarat di folder `dataset/letters/raw/` berdasarkan kelas:

```
dataset/letters/raw/
├── A/
│   ├── A_0001.avi
│   └── ...
├── B/
│   └── ...
└── Z/
    └── ...
```

### 3. Jalankan Pipeline Huruf 90 Frame

Pipeline berikut menggunakan [konfigurasi `configs/letters_90.yaml`](configs/letters_90.yaml), maksimal dua tangan, `input_size: 126`, dan panjang sequence 90 frame. Jalankan tahap-tahap secara berurutan dan jangan mencampur folder, config, atau checkpoint dari pipeline `letters` lama dengan pipeline `letters_90`.

#### 3.1 Normalisasi video ke 90 frame/30 FPS

Tahap ini membaca `dataset/letters/raw/`, kemudian membuat dataset baru di `dataset/letters/raw_90/` tanpa menimpa video sumber:

```powershell
python -m src.letters.normalize_letter_videos --input_dir dataset/letters/raw --output_dir dataset/letters/raw_90 --target_frames 90 --target_fps 30
```

Video yang sudah konsisten disalin tanpa re-encode. Video dengan jumlah frame atau FPS berbeda akan di-resampling menjadi tepat 90 frame pada 30 FPS. Hasil pemeriksaan disimpan dalam `dataset/letters/raw_90/normalization_report.csv`.

> Script tidak menimpa file yang sudah ada di `raw_90`. Jika ingin membuat ulang seluruh hasil normalisasi, pindahkan atau hapus output lama secara manual setelah memastikan data tersebut tidak lagi diperlukan.

#### 3.2 Ekstraksi landmark tangan

Ekstrak dua slot tangan tetap `[Left, Right]` dari semua video `raw_90`:

```powershell
python -m src.common.extract_landmarks --input_dir dataset/letters/raw_90 --output_dir dataset/letters/landmarks_90 --max_num_hands 2
```

Hasil setiap video berupa file `.npy` dengan bentuk `(90, 42, 3)` di `dataset/letters/landmarks_90/{kelas}/`.

Jika video sumber berubah dan landmark lama memang perlu dibuat ulang, tambahkan `--overwrite`:

```powershell
python -m src.common.extract_landmarks --input_dir dataset/letters/raw_90 --output_dir dataset/letters/landmarks_90 --max_num_hands 2 --overwrite
```

#### 3.3 Preprocessing

Normalisasi landmark, bagi data menjadi train/validation/test, augmentasi hanya pada train set, kemudian simpan tensor hasilnya di `processed_90`:

```powershell
python -m src.letters.preprocess --input_dir dataset/letters/landmarks_90 --output_dir dataset/letters/processed_90 --max_seq_length 90 --normalization wrist_relative --augment_factor 3 --test_size 0.15 --val_size 0.15 --random_seed 42
```

Output utama:

- `train_data.pt` dan `train_labels.pt`
- `val_data.pt` dan `val_labels.pt`
- `test_data.pt` dan `test_labels.pt`
- `label_encoder.json`

#### 3.4 Training model

Training mengambil seluruh path dan hyperparameter dari `configs/letters_90.yaml`:

```powershell
python -m src.letters.train --config configs/letters_90.yaml
```

Checkpoint terbaik disimpan di:

```text
outputs/letters_90/models/best_model.pth
```

Log dan kurva training disimpan di:

```text
outputs/letters_90/logs/training_log.csv
outputs/letters_90/figures/training_curves.png
```

#### 3.5 Evaluasi model

Gunakan config dan checkpoint dari pipeline yang sama:

```powershell
python -m src.letters.evaluate --model_path outputs/letters_90/models/best_model.pth --config configs/letters_90.yaml
```

Classification report, metrics, dan confusion matrix akan disimpan di `outputs/letters_90/results/` dan `outputs/letters_90/figures/`.

#### Ringkasan perintah lengkap

```powershell
python -m src.letters.normalize_letter_videos --input_dir dataset/letters/raw --output_dir dataset/letters/raw_90 --target_frames 90 --target_fps 30
python -m src.common.extract_landmarks --input_dir dataset/letters/raw_90 --output_dir dataset/letters/landmarks_90 --max_num_hands 2
python -m src.letters.preprocess --input_dir dataset/letters/landmarks_90 --output_dir dataset/letters/processed_90 --max_seq_length 90 --normalization wrist_relative --augment_factor 3 --test_size 0.15 --val_size 0.15 --random_seed 42
python -m src.letters.train --config configs/letters_90.yaml
python -m src.letters.evaluate --model_path outputs/letters_90/models/best_model.pth --config configs/letters_90.yaml
```

> Contoh di atas menggunakan sintaks PowerShell satu baris. Jangan menggunakan `\` sebagai penyambung baris di PowerShell; gunakan backtick (`` ` ``) atau tulis perintah dalam satu baris.

#### Alternatif pipeline huruf 60 frame

Untuk membuat model yang lebih ringan dan responsif saat digunakan melalui webcam,
jalankan pipeline yang sama dengan target 60 frame. Gunakan akhiran `_60` untuk
seluruh dataset, config, dan output agar tidak tercampur dengan pipeline 90 frame:

```powershell
python -m src.letters.normalize_letter_videos --input_dir dataset/letters/raw --output_dir dataset/letters/raw_60 --target_frames 60 --target_fps 30
python -m src.common.extract_landmarks --input_dir dataset/letters/raw_60 --output_dir dataset/letters/landmarks_60 --max_num_hands 2
python -m src.letters.preprocess --input_dir dataset/letters/landmarks_60 --output_dir dataset/letters/processed_60 --max_seq_length 60 --normalization wrist_relative --augment_factor 3 --test_size 0.15 --val_size 0.15 --random_seed 42
python -m src.letters.train --config configs/letters_60.yaml
python -m src.letters.evaluate --model_path outputs/letters_60/models/best_model.pth --config configs/letters_60.yaml
```

Pasangan file untuk inference model ini adalah `configs/letters_60.yaml` dan
`outputs/letters_60/models/best_model.pth`. Jangan menggunakan checkpoint 60
frame dengan config 90 frame, atau sebaliknya.

### 4. Test Prediksi Video Huruf

Prediksi satu video dari dataset 90 frame menggunakan config dan checkpoint yang sesuai:

```powershell
python -m src.letters.predict_video --video_path dataset/letters/raw_90/B/B_0001.avi --config configs/letters_90.yaml --model_path outputs/letters_90/models/best_model.pth
```

Gunakan ukuran window yang lebih kecil:

```powershell
python -m src.letters.predict_video --video_path dataset/letters/raw_90/B/B_0001.avi --config configs/letters_90.yaml --model_path outputs/letters_90/models/best_model.pth --display_width 640
```

Simpan hasil sebagai video MP4:

```powershell
python -m src.letters.predict_video --video_path dataset/letters/raw_90/B/B_0001.avi --config configs/letters_90.yaml --model_path outputs/letters_90/models/best_model.pth --output_path outputs/letters_90/predictions/B_0001_prediction.mp4
```

Untuk server/headless tanpa window OpenCV:

```powershell
python -m src.letters.predict_video --video_path dataset/letters/raw_90/B/B_0001.avi --config configs/letters_90.yaml --model_path outputs/letters_90/models/best_model.pth --output_path outputs/letters_90/predictions/B_0001_prediction.mp4 --no_display
```

Video menampilkan landmark tangan dan skeleton tubuh tanpa titik wajah. Landmark tubuh hanya untuk visualisasi; prediksi tetap menggunakan 126 fitur landmark tangan. Script memprediksi dari seluruh sequence video dan tidak membuat file landmark baru. Tekan **Q** atau **Esc** untuk menutup window.

### 5. Test Realtime dengan Webcam

Untuk penggunaan realtime, konfigurasi yang direkomendasikan adalah model 60
frame: gunakan `configs/letters_60.yaml` bersama checkpoint `outputs/letters_60`.
Model 90 frame tetap dapat digunakan untuk eksperimen, tetapi config dan checkpoint
harus selalu berasal dari pipeline yang sama. Hindari menjalankan webcam tanpa
kedua opsi tersebut karena nilai default script masih menunjuk ke pipeline
`letters` lama.

#### Konfigurasi realtime yang direkomendasikan

Sebagai konfigurasi awal realtime, model 60 frame dengan `motion_threshold=0.001`,
minimum 45 frame, dan voting 4 dari 5 prediksi memberikan kompromi yang masuk
akal antara kecepatan respons, jumlah padding, dan kestabilan kelas A serta W:

```powershell
python -m src.letters.predict_webcam --config configs/letters_60.yaml --model_path outputs/letters_60/models/best_model.pth --motion_threshold 0.001 --pause_frames 15 --min_recording_frames 45 --prediction_interval 3 --vote_window 5 --vote_ratio 0.8 --stable_confidence 0.8 --rearm_motion_frames 3
```

Saat berhasil memuat model yang digunakan pada evaluasi `letters_60`, terminal
akan menampilkan checkpoint dari **epoch 21**.

#### Perbedaan signifikan model 60 dan 90 frame

Kedua model menggunakan arsitektur dan jumlah parameter yang sama. Perbedaannya
adalah jumlah timestep yang dibaca BiLSTM pada setiap inference:

| Aspek | 60 frame | 90 frame |
|---|---:|---:|
| Panjang input maksimum | 60 timestep | 90 timestep |
| Perkiraan durasi pada 30 FPS | sekitar 2 detik | sekitar 3 detik |
| Akurasi test offline | 97,44% | 97,86% |
| Prediksi test benar | 228 dari 234 | 229 dari 234 |
| Epoch checkpoint terbaik | 21 | 37 |
| Beban pemrosesan sequence relatif | 1x | sekitar 1,5x |

Model 90 unggul satu sampel pada test offline, sehingga selisih 0,42 poin
persentase tersebut belum cukup untuk menyimpulkan bahwa model 90 selalu lebih
baik pada webcam. Dalam penggunaan realtime, model 60 dapat terasa lebih baik
karena lebih cepat diproses dan lebih sedikit memasukkan frame diam, getaran
landmark, atau gerakan menurunkan tangan setelah pose selesai.

`--min_recording_frames` menentukan kapan rolling prediction mulai dijalankan,
bukan panjang maksimum input. Jika prediksi dimulai sebelum buffer mencapai
panjang yang ditentukan config, script menambahkan padding nol. Contohnya:

| Pengaturan awal prediksi | Input model 60 | Input model 90 |
|---|---|---|
| `--min_recording_frames 30` | 30 frame nyata + 30 padding | 30 frame nyata + 60 padding |
| `--min_recording_frames 45` | 45 frame nyata + 15 padding | 45 frame nyata + 45 padding |

Model huruf saat ini belum menggunakan masking atau packed sequence, sehingga
padding tersebut tetap dibaca oleh LSTM. Karena itu, 45 frame direkomendasikan
untuk model 60 sebagai kompromi antara respons cepat dan input yang lebih lengkap.
Gunakan 30 jika latency menjadi prioritas utama dan hasilnya sudah stabil pada
perangkat serta pengguna yang dituju.

Model 90 masih berguna untuk gerakan yang lambat atau memiliki lintasan panjang,
misalnya huruf dinamis. Sebaliknya, model 60 biasanya cukup untuk pose statis dan
gerakan yang selesai dalam sekitar dua detik. Jika buffer lebih panjang daripada
`max_seq_length`, preprocessing saat ini mengambil frame pertama sebanyak panjang
config; bagian gerakan setelah frame ke-60 atau ke-90 tidak ikut diprediksi.

#### Rekam suara sendiri untuk hasil prediksi

Sistem menggunakan rekaman suara Anda sendiri, bukan Text-to-Speech sintetis. Instal dependency perekam melalui:

```powershell
pip install -r requirements.txt
```

Lihat daftar mikrofon dan perangkat output:

```powershell
python -m src.letters.record_letter_audio --list_devices
```

Jalankan perekam terpandu A–Z:

```powershell
python -m src.letters.record_letter_audio
```

Untuk setiap huruf, program akan:

1. Meminta Anda mengucapkan `Huruf A`, `Huruf B`, dan seterusnya.
2. Menunggu Enter untuk mulai merekam dan Enter lagi untuk berhenti.
3. Memutar preview rekaman.
4. Memberi pilihan **Simpan**, **Ulangi**, **Lewati**, atau **Quit**.

Rekaman disimpan sebagai mono PCM WAV 44,1 kHz/16-bit:

```text
assets/letters_audio/A.wav
assets/letters_audio/B.wav
...
assets/letters_audio/Z.wav
```

Jika ingin memilih mikrofon berdasarkan index dari `--list_devices`:

```powershell
python -m src.letters.record_letter_audio --device 1
```

Lanjutkan dari huruf tertentu atau rekam ulang file yang sudah ada:

```powershell
python -m src.letters.record_letter_audio --start_letter M
python -m src.letters.record_letter_audio --overwrite
```

Ketika hasil voting atau fallback terkunci, file WAV sesuai huruf diputar tepat satu kali secara asynchronous sehingga webcam tetap responsif. Gunakan folder rekaman default atau tentukan folder lain:

```powershell
python -m src.letters.predict_webcam --config configs/letters_60.yaml --model_path outputs/letters_60/models/best_model.pth --motion_threshold 0.001 --pause_frames 15 --min_recording_frames 45 --audio_dir assets/letters_audio
```

Jika file suatu huruf belum tersedia, terminal menampilkan warning dan prediksi tetap berjalan. Nonaktifkan seluruh audio dengan:

```powershell
python -m src.letters.predict_webcam --config configs/letters_60.yaml --model_path outputs/letters_60/models/best_model.pth --motion_threshold 0.001 --pause_frames 15 --min_recording_frames 45 --no_speech
```

Alur inference berjalan otomatis dan berulang:

1. `WAITING` — menunggu tangan mulai bergerak.
2. `RECORDING` — mengumpulkan minimal 45 frame, lalu menjalankan prediksi berkala setiap tiga frame.
3. Lima prediksi terbaru masuk ke voting. Hasil langsung dikunci jika sedikitnya empat vote sama dan rata-rata confidence kelas pemenang minimal 80%.
4. Jika voting belum stabil, `pause_frames` tetap digunakan sebagai fallback setelah tangan diam.
5. `RESULT` — hasil dan confidence ditampilkan tanpa pengguna harus menurunkan tangan.
6. `REARM` — setelah hasil selesai ditampilkan, cukup gerakkan tangan ke pose berikutnya selama tiga frame; pose diam yang sama tidak akan memicu hasil berulang.

#### Opsi webcam lain

Pilih kamera, perkecil window, dan atur durasi tampilan hasil:

```powershell
python -m src.letters.predict_webcam --config configs/letters_60.yaml --model_path outputs/letters_60/models/best_model.pth --camera_index 0 --display_width 640 --motion_threshold 0.001 --pause_frames 15 --min_recording_frames 45 --result_frames 45
```

Tampilkan hasil hanya jika confidence mencapai minimal 70%:

```powershell
python -m src.letters.predict_webcam --config configs/letters_60.yaml --model_path outputs/letters_60/models/best_model.pth --motion_threshold 0.001 --pause_frames 15 --min_recording_frames 45 --min_confidence 0.70
```

Parameter penting:

| Parameter | Rekomendasi | Fungsi |
|---|---:|---|
| `--camera_index` | `0` | Indeks kamera OpenCV |
| `--motion_threshold` | `0.001` | Ambang perpindahan landmark yang dianggap gerakan |
| `--pause_frames` | `15` | Frame diam sebelum fallback prediksi dijalankan |
| `--min_recording_frames` | `45` | Sequence minimum model 60 sebelum rolling prediction; mengurangi padding dan prediksi terlalu dini |
| `--prediction_interval` | `3` | Jarak frame antar-inferensi selama recording |
| `--vote_window` | `5` | Jumlah prediksi terbaru yang ikut voting |
| `--vote_ratio` | `0.8` | Proporsi vote kelas pemenang untuk mengunci hasil |
| `--stable_confidence` | `0.8` | Rata-rata confidence kelas pemenang untuk mengunci hasil |
| `--rearm_motion_frames` | `3` | Gerakan berturut-turut untuk mulai huruf berikutnya |
| `--result_frames` | `45` | Lama hasil terkunci ditampilkan sebelum masuk `REARM` |
| `--display_width` | `960` atau `640` | Lebar maksimum window |
| `--min_confidence` | `0.0`–`1.0` | Confidence minimum untuk memberi warna hasil akhir |
| `--audio_dir` | `assets/letters_audio` | Folder rekaman suara `A.wav`–`Z.wav` |
| `--no_speech` | nonaktif | Jalankan prediksi tanpa memutar rekaman suara |

`pause_frames` sekarang berfungsi sebagai fallback jika voting belum stabil. Untuk
model 60, nilai 45 direkomendasikan agar prediksi tidak dimulai terlalu dini.
Nilai 30 dapat digunakan untuk respons lebih cepat, tetapi harus diuji ulang pada
A, W, dan huruf dinamis karena input awal mengandung lebih banyak padding. Jika
ingin hasil terkunci lebih cepat, parameter lain yang dapat diuji adalah
`prediction_interval`, `vote_window`, atau `vote_ratio`; nilai yang terlalu longgar
dapat membuat prediksi berkedip atau salah terkunci.

Lakukan gerakan secara jelas dan tahan pose akhir. Ketika vote sudah stabil, hasil akan muncul meskipun tangan masih diangkat. Setelah hasil selesai ditampilkan, langsung gerakkan tangan ke pose baru tanpa perlu menurunkannya. Tekan **Q** atau **Esc** untuk keluar.

Skeleton tubuh tanpa titik wajah hanya untuk visualisasi. Model tetap menggunakan landmark tangan. Mode webcam tidak merekam atau menyimpan video. Opsi `--mirror` membalik tampilan sekaligus input sebelum ditampilkan dan tidak direkomendasikan jika orientasi dataset training tidak dicerminkan.

#### Kirim hasil prediksi ke app mobile web

Jalankan backend `app-mobile` pada port `3001`, lalu tambahkan opsi berikut ke
perintah webcam:

```powershell
python -m src.letters.predict_webcam --config configs/letters_60.yaml --model_path outputs/letters_60/models/best_model.pth --motion_threshold 0.001 --pause_frames 15 --min_recording_frames 45 --prediction_interval 3 --vote_window 5 --vote_ratio 0.8 --stable_confidence 0.8 --rearm_motion_frames 3 --server_url http://localhost:3001 --room_id demo-ta --no_speech
```

Model akan:

1. Mengirim label dan confidence ketika hasil prediksi terkunci.
2. Menampilkan teks dan suara melalui frontend yang terhubung ke room `demo-ta`.
3. Tetap menampilkan webcam hanya di window lokal; frame kamera tidak dikirim ke
   backend atau perangkat mobile.

Gunakan URL backend lokal pada `--server_url`. Cloudflare Tunnel cukup dijalankan
untuk backend; model tidak perlu mengirim data keluar melalui URL tunnel.

Opsi integrasi:

| Parameter | Default | Fungsi |
|---|---:|---|
| `--server_url` | kosong | URL backend lokal; integrasi nonaktif jika kosong |
| `--room_id` | `demo-ta` | Room tujuan hasil prediksi |
| `--model_api_key` | kosong | API key opsional, atau gunakan environment `MODEL_API_KEY` |

> **Catatan PowerShell:** karakter `\` bukan penyambung baris. Tulis perintah dalam satu baris seperti contoh di atas, atau gunakan backtick (`` ` ``) pada akhir setiap baris tanpa spasi setelahnya.

## 📁 Struktur Project

- `configs/letters_60.yaml` — konfigurasi pipeline huruf 60 frame yang direkomendasikan untuk realtime
- `configs/letters_90.yaml` — konfigurasi pipeline huruf 90 frame
- `dataset/letters/raw/` — video sumber per kelas
- `dataset/letters/raw_60/` — video hasil normalisasi 60 frame/30 FPS
- `dataset/letters/landmarks_60/` — hasil ekstraksi landmark 60 frame `.npy`
- `dataset/letters/processed_60/` — tensor train/val/test model 60 dan label encoder
- `outputs/letters_60/` — checkpoint, log, figures, dan hasil evaluasi model 60
- `dataset/letters/raw_90/` — video hasil normalisasi 90 frame/30 FPS
- `dataset/letters/landmarks_90/` — hasil ekstraksi landmark `.npy`
- `dataset/letters/processed_90/` — tensor train/val/test dan label encoder
- `outputs/letters_90/models/` — checkpoint model hasil training
- `outputs/letters_90/logs/` — log training
- `outputs/letters_90/figures/` — kurva training dan confusion matrix
- `outputs/letters_90/results/` — metrics dan classification report
- `outputs/letters_90/predictions/` — lokasi yang disarankan untuk video hasil prediksi
- `assets/letters_audio/` — rekaman suara pengguna `A.wav`–`Z.wav`
- `src/letters/record_letter_audio.py` — perekam suara A–Z terpandu
- `src/letters/` — normalisasi, preprocessing, model, training, evaluasi, dan inference huruf
- `src/common/` — ekstraksi landmark dan utilitas bersama

Folder tanpa akhiran `_60` atau `_90` merupakan pipeline/dataset sebelumnya.
Pastikan config, processed data, checkpoint, dan perintah inference selalu berasal
dari pipeline yang sama.

Lihat [WORKFLOW.md](WORKFLOW.md) untuk dokumentasi lengkap setiap tahap dan opsi prediksi video.

## 🛠️ Teknologi

- **PyTorch** — Deep Learning Framework
- **MediaPipe** — Hand Landmark Detection
- **OpenCV** — Video Processing dan visualisasi prediksi
- **scikit-learn** — Data Splitting & Metrics

## 📄 Lisensi

MIT License
