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

### 3. Jalankan Pipeline Huruf

Konfigurasi huruf saat ini menggunakan maksimal dua tangan (`input_size: 126`). Nilai `--max_num_hands`, jumlah fitur, dan checkpoint harus selalu sama dengan konfigurasi yang digunakan saat training.

```bash
# Ekstraksi landmark (dua slot tangan sesuai configs/letters.yaml)
python src/extract_landmarks.py --input_dir dataset/letters/raw --output_dir dataset/letters/landmarks --max_num_hands 2

# Preprocessing
python src/preprocess.py --input_dir dataset/letters/landmarks --output_dir dataset/letters/processed --max_seq_length 90 --normalization wrist_relative --augment_factor 3

# Training
python src/train.py --config configs/letters.yaml

# Evaluasi
python src/evaluate.py --model_path outputs/letters/models/best_model.pth --config configs/letters.yaml
```

### 4. Test Prediksi Video Huruf

Prediksi satu video dataset dan tampilkan video beranotasi. Video menampilkan titik serta garis landmark tangan dan skeleton tubuh tanpa titik wajah dari MediaPipe Pose. Landmark tubuh hanya untuk visualisasi; prediksi model tetap menggunakan landmark tangan. Teks hasil baru ditampilkan setelah sistem mendeteksi gerakan tangan lalu jeda. Setelah pemutaran, terminal menanyakan apakah video ingin disimpan; Enter pada lokasi tujuan memakai `outputs/letters/predictions/`. Ukuran window default memiliki lebar maksimum 960 piksel. Tekan **Q** atau **Esc** untuk menutup window.

```bash
python src/predict_video.py --video_path dataset/letters/raw/B/B_0001.avi

# Gunakan ukuran window yang lebih kecil
python src/predict_video.py --video_path dataset/letters/raw/B/B_0001.avi --display_width 640
```

Simpan hasil sebagai video MP4:

```bash
python src/predict_video.py \
  --video_path dataset/letters/raw/B/B_0001.avi \
  --output_path outputs/letters/predictions/B_0001_prediction.mp4
```

Untuk server/headless tanpa window OpenCV:

```bash
python src/predict_video.py \
  --video_path dataset/letters/raw/B/B_0001.avi \
  --output_path outputs/letters/predictions/B_0001_prediction.mp4 \
  --no_display
```

Script memprediksi dari **seluruh sequence video**, lalu menampilkan kelas huruf dan confidence pada setiap frame video hasil. Script ini tidak membuat file landmark baru.

## 📁 Struktur Project

- `configs/letters.yaml` — konfigurasi pipeline huruf
- `dataset/letters/raw/` — video mentah per kelas
- `dataset/letters/landmarks/` — hasil ekstraksi `.npy`
- `dataset/letters/processed/` — tensor train/val/test dan label encoder
- `outputs/letters/models/` — checkpoint model
- `outputs/letters/predictions/` — lokasi yang disarankan untuk video hasil prediksi
- `src/predict_video.py` — inference dan visualisasi satu video

Lihat [WORKFLOW.md](WORKFLOW.md) untuk dokumentasi lengkap setiap tahap dan opsi prediksi video.

## 🛠️ Teknologi

- **PyTorch** — Deep Learning Framework
- **MediaPipe** — Hand Landmark Detection
- **OpenCV** — Video Processing dan visualisasi prediksi
- **scikit-learn** — Data Splitting & Metrics

## 📄 Lisensi

MIT License
