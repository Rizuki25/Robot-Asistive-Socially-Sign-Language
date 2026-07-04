# 🤟 Sign Language Recognition - BiLSTM Pipeline

## Gambaran Umum Project

Project ini membangun sistem pengenalan bahasa isyarat menggunakan arsitektur **Bidirectional LSTM (BiLSTM)** dengan **PyTorch**. Pipeline dimulai dari video dataset mentah hingga model yang siap digunakan untuk prediksi.

---

## Alur Kerja (Workflow Pipeline)

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐    ┌────────────────┐
│  Ekstraksi   │───▶│  Preprocessing   │───▶│  Pemodelan &        │───▶│   Evaluasi     │
│  Landmark    │    │  Data            │    │  Pelatihan BiLSTM   │    │   Model        │
└──────────────┘    └──────────────────┘    └─────────────────────┘    └────────────────┘
```

---

## Tahap 1: Ekstraksi Landmark (`src/extract_landmarks.py`)

### Tujuan
Mengekstrak 21 titik landmark tangan dari setiap frame video menggunakan **MediaPipe Hands**.

### Input
- Video dataset dalam format `.mp4` / `.avi` yang tersimpan di `dataset/raw/`
- Struktur folder: `dataset/raw/{nama_kelas}/video1.mp4`

### Proses
1. Baca setiap video frame-by-frame menggunakan OpenCV
2. Deteksi tangan menggunakan MediaPipe Hands
3. Ekstrak 21 titik landmark (x, y, z) per frame
4. Simpan sequence landmark per video sebagai file `.npy`

### Output
- File `.npy` berisi array landmark per video
- Disimpan di `dataset/landmarks/{nama_kelas}/video1.npy`
- Shape per file: `(num_frames, 21, 3)` → 21 landmark × 3 koordinat (x, y, z)

### Cara Menjalankan
```bash
python src/extract_landmarks.py --input_dir dataset/raw --output_dir dataset/landmarks
```

---

## Tahap 2: Preprocessing Data (`src/preprocess.py`)

### Tujuan
Mempersiapkan data landmark agar siap dilatih oleh model BiLSTM.

### Input
- File `.npy` landmark dari Tahap 1 di `dataset/landmarks/`

### Proses
1. **Normalisasi Data**: Normalisasi koordinat landmark (min-max atau z-score)
2. **Penyusunan Sequence (Time Series)**: Padding/truncating sequence agar panjang seragam (`MAX_SEQ_LENGTH`)
3. **Flatten Landmark**: Reshape dari `(seq_len, 21, 3)` menjadi `(seq_len, 63)` sebagai input BiLSTM
4. **Label Encoding**: Konversi nama kelas menjadi integer label
5. **Pembagian Data**: Split menjadi Train/Validation/Test set (misal: 70/15/15)

### Output
- File `.pt` (PyTorch tensors) untuk train, val, dan test set
- Disimpan di `dataset/processed/`
  - `train_data.pt`, `train_labels.pt`
  - `val_data.pt`, `val_labels.pt`
  - `test_data.pt`, `test_labels.pt`
  - `label_encoder.json` (mapping label ↔ nama kelas)

### Cara Menjalankan
```bash
python src/preprocess.py --input_dir dataset/landmarks --output_dir dataset/processed --max_seq_length 50 --test_size 0.15 --val_size 0.15
```

---

## Tahap 3: Pemodelan & Pelatihan BiLSTM (`src/train.py`)

### Tujuan
Membangun dan melatih model BiLSTM untuk klasifikasi bahasa isyarat.

### Input
- Data `.pt` yang telah diproses dari Tahap 2

### Arsitektur Model (`src/model.py`)
```
Input (batch, seq_len, 63)
        │
  ┌─────▼─────┐
  │  BiLSTM    │  ← Bidirectional LSTM layer(s)
  │  Layer(s)  │     hidden_size, num_layers, dropout
  └─────┬─────┘
        │
  ┌─────▼─────┐
  │  Dropout   │
  └─────┬─────┘
        │
  ┌─────▼─────┐
  │ Fully      │  ← Linear layer
  │ Connected  │     input: hidden_size * 2 (bidirectional)
  └─────┬─────┘     output: num_classes
        │
  ┌─────▼─────┐
  │  Softmax   │
  └───────────┘
```

### Proses Pelatihan
1. Load data dari `dataset/processed/`
2. Buat PyTorch `DataLoader` untuk train, val, dan test
3. Definisikan model BiLSTM, loss function (`CrossEntropyLoss`), dan optimizer (`Adam`)
4. Training loop dengan early stopping dan learning rate scheduler
5. Simpan model terbaik berdasarkan validation loss

### Hyperparameter (dikonfigurasi di `configs/config.yaml`)
- `hidden_size`: Ukuran hidden state LSTM (default: 128)
- `num_layers`: Jumlah layer LSTM (default: 2)
- `dropout`: Dropout rate (default: 0.3)
- `learning_rate`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 32)
- `epochs`: Maksimum epoch (default: 100)
- `patience`: Early stopping patience (default: 10)

### Output
- Model terlatih: `outputs/models/best_model.pth`
- Training log: `outputs/logs/training_log.csv`
- Loss & accuracy curves: `outputs/figures/`

### Cara Menjalankan
```bash
python src/train.py --config configs/config.yaml
```

---

## Tahap 4: Evaluasi Model (`src/evaluate.py`)

### Tujuan
Mengevaluasi performa model pada test set.

### Input
- Model terlatih dari `outputs/models/best_model.pth`
- Test data dari `dataset/processed/`

### Metrik Evaluasi
1. **Accuracy** - Akurasi keseluruhan
2. **Precision** - Ketepatan per kelas
3. **Recall** - Sensitivitas per kelas
4. **F1-Score** - Harmonic mean dari precision dan recall
5. **Confusion Matrix** - Visualisasi prediksi vs aktual

### Output
- Classification report: `outputs/results/classification_report.txt`
- Confusion matrix plot: `outputs/figures/confusion_matrix.png`
- Detailed metrics: `outputs/results/evaluation_metrics.json`

### Cara Menjalankan
```bash
python src/evaluate.py --model_path outputs/models/best_model.pth --config configs/config.yaml
```

---

## Struktur Folder Project

```
SignLanguage_BiLSTM/
│
├── WORKFLOW.md                  # ← File ini (dokumentasi alur kerja)
├── README.md                    # Dokumentasi umum project
├── requirements.txt             # Daftar dependensi Python
├── .gitignore                   # File yang diabaikan Git
│
├── configs/
│   └── config.yaml              # Konfigurasi hyperparameter & path
│
├── dataset/
│   ├── raw/                     # Video mentah (dikelompokkan per kelas)
│   │   ├── halo/
│   │   │   ├── video001.mp4
│   │   │   └── ...
│   │   ├── terima_kasih/
│   │   └── ...
│   │
│   ├── landmarks/               # Hasil ekstraksi landmark (.npy)
│   │   ├── halo/
│   │   │   ├── video001.npy
│   │   │   └── ...
│   │   └── ...
│   │
│   └── processed/               # Data siap latih (train/val/test .pt)
│       ├── train_data.pt
│       ├── train_labels.pt
│       ├── val_data.pt
│       ├── val_labels.pt
│       ├── test_data.pt
│       ├── test_labels.pt
│       └── label_encoder.json
│
├── src/
│   ├── __init__.py
│   ├── extract_landmarks.py     # Tahap 1: Ekstraksi landmark MediaPipe
│   ├── preprocess.py            # Tahap 2: Preprocessing & split data
│   ├── model.py                 # Definisi arsitektur BiLSTM
│   ├── dataset_loader.py        # PyTorch Dataset & DataLoader
│   ├── train.py                 # Tahap 3: Training loop
│   ├── evaluate.py              # Tahap 4: Evaluasi model
│   └── utils.py                 # Fungsi utilitas umum
│
├── notebooks/
│   └── exploration.ipynb        # Notebook eksplorasi & visualisasi
│
└── outputs/
    ├── models/                  # Model tersimpan (.pth)
    ├── logs/                    # Training logs
    ├── figures/                 # Grafik & visualisasi
    └── results/                 # Hasil evaluasi
```

---

## Urutan Eksekusi

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Ekstraksi landmark dari video dataset
python src/extract_landmarks.py --input_dir dataset/raw --output_dir dataset/landmarks

# 3. Preprocessing data (normalisasi, padding, split)
python src/preprocess.py --input_dir dataset/landmarks --output_dir dataset/processed

# 4. Training model BiLSTM
python src/train.py --config configs/config.yaml

# 5. Evaluasi model
python src/evaluate.py --model_path outputs/models/best_model.pth --config configs/config.yaml
```

---

## Teknologi yang Digunakan

| Teknologi | Kegunaan |
|-----------|----------|
| **Python 3.8+** | Bahasa pemrograman utama |
| **PyTorch** | Framework deep learning |
| **MediaPipe** | Ekstraksi landmark tangan |
| **OpenCV** | Pembacaan & pemrosesan video |
| **NumPy** | Operasi array & numerik |
| **scikit-learn** | Split data & metrik evaluasi |
| **matplotlib / seaborn** | Visualisasi grafik |
| **PyYAML** | Parsing file konfigurasi |
| **tqdm** | Progress bar |

---

## Catatan Penting untuk AI Agent

1. **Dataset sudah tersedia** — Tidak perlu mengunduh atau membuat dataset. Cukup tempatkan video di `dataset/raw/{nama_kelas}/`.
2. **Framework: PyTorch** — Semua pemodelan dan training menggunakan PyTorch, **BUKAN TensorFlow/Keras**.
3. **MediaPipe Hands** — Gunakan `mediapipe.solutions.hands` untuk ekstraksi 21 titik landmark.
4. **Koordinat Landmark** — Setiap landmark memiliki 3 nilai: `(x, y, z)`, total 63 fitur per frame.
5. **Sequence Length** — Perlu padding/truncating agar semua video memiliki panjang sequence yang sama.
6. **BiLSTM** — Model menggunakan `bidirectional=True` pada `nn.LSTM` PyTorch.
7. **Data Split** — Train/Val/Test disimpan dalam folder `dataset/processed/`, bukan folder terpisah.
8. **Reproducibility** — Selalu set random seed untuk reproducibility.
