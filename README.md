# 🤟 Sign Language Recognition using BiLSTM

Sistem pengenalan bahasa isyarat menggunakan **Bidirectional LSTM (BiLSTM)** dengan **PyTorch**.

## 📋 Deskripsi

Project ini menggunakan **MediaPipe Hands** untuk mengekstrak 21 titik landmark tangan dari video bahasa isyarat, kemudian melatih model **BiLSTM** untuk mengklasifikasikan gestur tangan.

## 🏗️ Arsitektur Pipeline

```
Video Dataset → Ekstraksi Landmark → Preprocessing → Training BiLSTM → Evaluasi
                  (MediaPipe)         (Normalisasi)    (PyTorch)       (Metrics)
```

## 🚀 Quick Start

### 1. Install Dependensi
```bash
pip install -r requirements.txt
```

### 2. Siapkan Dataset
Tempatkan video bahasa isyarat di folder `dataset/raw/` dengan struktur:
```
dataset/raw/
├── kelas_1/
│   ├── video001.mp4
│   └── ...
├── kelas_2/
│   └── ...
└── ...
```

### 3. Jalankan Pipeline
```bash
# Ekstraksi landmark
python src/extract_landmarks.py --input_dir dataset/raw --output_dir dataset/landmarks

# Preprocessing
python src/preprocess.py --input_dir dataset/landmarks --output_dir dataset/processed

# Training
python src/train.py --config configs/config.yaml

# Evaluasi
python src/evaluate.py --model_path outputs/models/best_model.pth --config configs/config.yaml
```

## 📁 Struktur Project

Lihat [WORKFLOW.md](WORKFLOW.md) untuk dokumentasi lengkap alur kerja dan penjelasan setiap tahapan.

## 🛠️ Teknologi

- **PyTorch** — Deep Learning Framework
- **MediaPipe** — Hand Landmark Detection
- **OpenCV** — Video Processing
- **scikit-learn** — Data Splitting & Metrics

## 📄 Lisensi

MIT License
