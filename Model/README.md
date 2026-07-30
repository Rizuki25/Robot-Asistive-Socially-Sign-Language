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

Gunakan `configs/letters_90.yaml` bersama checkpoint `outputs/letters_90`. Hindari menjalankan webcam tanpa kedua opsi ini karena nilai default script masih menunjuk ke pipeline `letters` lama.

#### Konfigurasi realtime yang direkomendasikan

Berdasarkan pengujian realtime, `motion_threshold=0.001` dan `pause_frames=15` memberikan kompromi yang baik antara kecepatan respons dan kestabilan kelas A serta W:

```powershell
python -m src.letters.predict_webcam --config configs/letters_90.yaml --model_path outputs/letters_90/models/best_model.pth --motion_threshold 0.001 --pause_frames 15
```

Saat berhasil memuat model yang digunakan pada evaluasi `letters_90`, terminal akan menampilkan checkpoint dari **epoch 37**.

Alur inference berjalan otomatis dan berulang:

1. `WAITING` — menunggu tangan mulai bergerak.
2. `RECORDING` — mengumpulkan sequence landmark.
3. Setelah tangan dianggap diam selama `pause_frames`, model melakukan prediksi.
4. `RESULT` — hasil dan confidence ditampilkan, lalu sistem kembali menunggu gerakan berikutnya.

#### Opsi webcam lain

Pilih kamera, perkecil window, dan atur durasi tampilan hasil:

```powershell
python -m src.letters.predict_webcam --config configs/letters_90.yaml --model_path outputs/letters_90/models/best_model.pth --camera_index 0 --display_width 640 --motion_threshold 0.001 --pause_frames 15 --result_frames 45
```

Tampilkan hasil hanya jika confidence mencapai minimal 70%:

```powershell
python -m src.letters.predict_webcam --config configs/letters_90.yaml --model_path outputs/letters_90/models/best_model.pth --motion_threshold 0.001 --pause_frames 15 --min_confidence 0.70
```

Parameter penting:

| Parameter | Rekomendasi | Fungsi |
|---|---:|---|
| `--camera_index` | `0` | Indeks kamera OpenCV |
| `--motion_threshold` | `0.001` | Ambang perpindahan landmark yang dianggap gerakan |
| `--pause_frames` | `15` | Jumlah frame diam berturut-turut sebelum prediksi |
| `--result_frames` | `45` | Lama hasil ditampilkan sebelum reset |
| `--display_width` | `960` atau `640` | Lebar maksimum window |
| `--min_confidence` | `0.0`–`1.0` | Confidence minimum untuk menandai hasil |

`pause_frames` dapat disesuaikan berdasarkan FPS dan kestabilan prediksi:

- `15` frame: rekomendasi; A dan W lebih stabil.
- `12` frame atau kurang: hasil lebih cepat, tetapi pada pengujian realtime A dan W mulai salah kembali.
- Lebih dari `15` frame: lebih stabil, tetapi pengguna harus menahan pose lebih lama.

Lakukan gerakan secara jelas, tahan pose akhir sebentar, dan jangan langsung menurunkan tangan. Tekan **Q** atau **Esc** untuk keluar.

Skeleton tubuh tanpa titik wajah hanya untuk visualisasi. Model tetap menggunakan landmark tangan. Mode webcam tidak merekam atau menyimpan video. Opsi `--mirror` membalik tampilan sekaligus input sebelum ditampilkan dan tidak direkomendasikan jika orientasi dataset training tidak dicerminkan.

> **Catatan PowerShell:** karakter `\` bukan penyambung baris. Tulis perintah dalam satu baris seperti contoh di atas, atau gunakan backtick (`` ` ``) pada akhir setiap baris tanpa spasi setelahnya.

## 📁 Struktur Project

- `configs/letters_90.yaml` — konfigurasi pipeline huruf 90 frame
- `dataset/letters/raw/` — video sumber per kelas
- `dataset/letters/raw_90/` — video hasil normalisasi 90 frame/30 FPS
- `dataset/letters/landmarks_90/` — hasil ekstraksi landmark `.npy`
- `dataset/letters/processed_90/` — tensor train/val/test dan label encoder
- `outputs/letters_90/models/` — checkpoint model hasil training
- `outputs/letters_90/logs/` — log training
- `outputs/letters_90/figures/` — kurva training dan confusion matrix
- `outputs/letters_90/results/` — metrics dan classification report
- `outputs/letters_90/predictions/` — lokasi yang disarankan untuk video hasil prediksi
- `src/letters/` — normalisasi, preprocessing, model, training, evaluasi, dan inference huruf
- `src/common/` — ekstraksi landmark dan utilitas bersama

Folder tanpa akhiran `_90` merupakan pipeline/dataset sebelumnya. Pastikan config, processed data, checkpoint, dan perintah inference selalu berasal dari pipeline yang sama.

Lihat [WORKFLOW.md](WORKFLOW.md) untuk dokumentasi lengkap setiap tahap dan opsi prediksi video.

## 🛠️ Teknologi

- **PyTorch** — Deep Learning Framework
- **MediaPipe** — Hand Landmark Detection
- **OpenCV** — Video Processing dan visualisasi prediksi
- **scikit-learn** — Data Splitting & Metrics

## 📄 Lisensi

MIT License
