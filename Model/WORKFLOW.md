# 🤟 Sign Language Recognition - BiLSTM Pipeline

## Gambaran Umum Project

Project ini membangun sistem pengenalan bahasa isyarat menggunakan arsitektur **Bidirectional LSTM (BiLSTM)** dengan **PyTorch**. Pipeline dimulai dari video dataset mentah hingga model yang siap digunakan untuk prediksi.

Project ini terdiri dari **dua model independen** yang berbagi satu codebase (`src/`) yang sama:

| Task | Kelas | Tangan | Config | Status |
|------|-------|--------|--------|--------|
| **Huruf** (fingerspelling) | A-Z (26 kelas) | Maks. 2 tangan (sesuai config) | `configs/letters.yaml` | ✅ Dataset & model sudah ada |
| **Kata** (isyarat kata) | TBD | Bisa 2 tangan | `configs/words.yaml` | ⏳ Dataset belum tersedia |

Kedua task punya `dataset/`, `outputs/`, dan `config.yaml` sendiri-sendiri (lihat [Struktur Folder](#struktur-folder-project)) — label encoder-nya terpisah total, jadi kelas huruf dan kata **tidak pernah tercampur** dalam satu output layer. Script di `src/` sama persis untuk keduanya; yang membedakan cuma argumen CLI/`--config` yang dipakai.

---

## Alur Kerja (Workflow Pipeline)

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐    ┌────────────────┐
│  Ekstraksi   │───▶│  Preprocessing   │───▶│  Pemodelan &        │───▶│   Evaluasi     │
│  Landmark    │    │  Data            │    │  Pelatihan BiLSTM   │    │   Model        │
└──────────────┘    └──────────────────┘    └─────────────────────┘    └────────────────┘
```

Alur ini dijalankan **terpisah** untuk task huruf dan task kata (dataset/output berbeda folder).

---

## Tahap 1: Ekstraksi Landmark (`src/common/extract_landmarks.py`)

### Tujuan
Mengekstrak titik landmark tangan dari setiap frame video menggunakan **MediaPipe Hands**.

### Input
- Video dataset dalam format `.mp4` / `.avi` / `.mov` / `.mkv` / `.wmv`
- Struktur folder: `dataset/{task}/raw/{nama_kelas}/video1.mp4`
  - Huruf: `dataset/letters/raw/{A,B,...,Z}/`
  - Kata: `dataset/words/raw/{nama_kata}/`

### Proses
1. Baca setiap video frame-by-frame menggunakan OpenCV
2. Deteksi tangan menggunakan MediaPipe Hands (`max_num_hands` bisa 1 atau 2)
3. Kalau `max_num_hands=2`: tiap tangan ditaruh di slot tetap **[Left, Right]** berdasarkan `results.multi_handedness` (konsisten antar frame); tangan yang tak terdeteksi di suatu frame diisi nol, bukan di-skip
4. Ekstrak 21 titik landmark (x, y, z) per tangan per frame
5. Simpan sequence landmark per video sebagai file `.npy`
6. **Incremental by default**: video yang `.npy`-nya sudah ada di `output_dir` otomatis di-skip (pakai `--overwrite` untuk paksa ekstrak ulang semua)

### Output
- File `.npy` berisi array landmark per video, disimpan di `dataset/{task}/landmarks/{nama_kelas}/video1.npy`
- Shape per file: `(num_frames, 21 * max_num_hands, 3)`
  - `max_num_hands=1` → `(num_frames, 21, 3)`
  - `max_num_hands=2` (config huruf saat ini dan kata) → `(num_frames, 42, 3)`

### Cara Menjalankan
```bash
# Huruf (ikuti config letters.yaml: maksimal 2 tangan)
python -m src.common.extract_landmarks --input_dir dataset/letters/raw --output_dir dataset/letters/landmarks --max_num_hands 2

# Kata (2 tangan)
python -m src.common.extract_landmarks --input_dir dataset/words/raw --output_dir dataset/words/landmarks --max_num_hands 2

# Paksa ekstrak ulang semua video (bukan cuma yang baru)
python -m src.common.extract_landmarks --input_dir dataset/letters/raw --output_dir dataset/letters/landmarks --overwrite
```

---

## Tahap 2: Preprocessing Data

### Tujuan
Mempersiapkan data landmark agar siap dilatih oleh model BiLSTM.

### Input
- File `.npy` landmark dari Tahap 1 di `dataset/{task}/landmarks/`

### Proses
1. **Split index dulu** (stratified per kelas, train/val/test) — dilakukan **sebelum** normalisasi & augmentasi supaya val/test tidak pernah tersentuh data sintetis
2. **Normalisasi** (`--normalization wrist_relative`, default): tiap tangan di-center ke wrist-nya sendiri (translation-invariant) lalu diskalakan dengan jarak wrist→pangkal jari tengah (scale-invariant). Diterapkan per-blok 21 landmark, jadi otomatis mendukung config 1 tangan maupun 2 tangan. Opsi lama `minmax`/`zscore` masih tersedia untuk eksperimen.
3. **Augmentasi (hanya pada train split)** — `--augment_factor N` (default 3) menghasilkan N salinan teraugmentasi per sample training lewat kombinasi rotasi kecil, scale jitter, gaussian noise, dan time-warp (variasi kecepatan). Val/test **tidak** diaugmentasi.
4. **Padding/Truncating**: sequence disamakan panjangnya ke `--max_seq_length` (cek dulu distribusi panjang video asli sebelum menentukan angka ini — jangan asal pakai default)
5. **Flatten Landmark**: reshape dari `(seq_len, 21*num_hands, 3)` menjadi `(seq_len, 63*num_hands)` sebagai input BiLSTM
6. **Label Encoding**: konversi nama kelas menjadi integer label

### Output
- File `.pt` (PyTorch tensors) untuk train, val, dan test set, disimpan di `dataset/{task}/processed/`
  - `train_data.pt`, `train_labels.pt`
  - `val_data.pt`, `val_labels.pt`
  - `test_data.pt`, `test_labels.pt`
  - `label_encoder.json` (mapping label ↔ nama kelas)

### Cara Menjalankan
```bash
# Huruf
python -m src.letters.preprocess --input_dir dataset/letters/landmarks --output_dir dataset/letters/processed --max_seq_length 90 --normalization wrist_relative --augment_factor 3

# Kata (setelah dataset kata tersedia — cek dulu statistik panjang video sebelum set max_seq_length)
python -m src.words.preprocess --config configs/words_motion.yaml
```

---

## Tahap 3: Pemodelan & Pelatihan BiLSTM

### Tujuan
Membangun dan melatih model BiLSTM untuk klasifikasi bahasa isyarat.

### Input
- Data `.pt` yang telah diproses dari Tahap 2

### Arsitektur Model (`src/letters/model.py` dan `src/words/model.py`)
```
Input (batch, seq_len, input_size)   # input_size = 63 (1 tangan) atau 126 (2 tangan; config huruf saat ini)
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
1. Load data dari `dataset/{task}/processed/`
2. Buat PyTorch `DataLoader` untuk train, val, dan test
3. Definisikan model BiLSTM, loss function (`CrossEntropyLoss`), dan optimizer (`Adam`)
4. Training loop dengan early stopping dan learning rate scheduler
5. Simpan model terbaik berdasarkan validation loss

### Hyperparameter (dikonfigurasi di `configs/letters.yaml` / `configs/words.yaml`)
- `input_size`: 63 untuk 1 tangan atau 126 untuk 2 tangan. Config huruf saat ini memakai 126 — harus cocok dengan `max_num_hands` yang dipakai saat ekstraksi
- `hidden_size`: Ukuran hidden state LSTM (default: 128)
- `num_layers`: Jumlah layer LSTM (default: 2)
- `dropout`: Dropout rate (default: 0.5)
- `weight_decay`: L2 regularization (default: 0.0005)
- `learning_rate`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 32)
- `epochs`: Maksimum epoch (default: 100)
- `patience`: Early stopping patience (default: 10)

### Output
- Model terlatih: `outputs/{task}/models/best_model.pth`
- Training log: `outputs/{task}/logs/training_log.csv`
- Loss & accuracy curves: `outputs/{task}/figures/`

### Cara Menjalankan
```bash
# Huruf
python -m src.letters.train --config configs/letters.yaml

# Kata
python -m src.words.train --config configs/words_motion.yaml
```

---

## Tahap 4: Evaluasi Model

### Tujuan
Mengevaluasi performa model pada test set.

### Input
- Model terlatih dari `outputs/{task}/models/best_model.pth`
- Test data dari `dataset/{task}/processed/`

### Metrik Evaluasi
1. **Accuracy** - Akurasi keseluruhan
2. **Precision** - Ketepatan per kelas
3. **Recall** - Sensitivitas per kelas
4. **F1-Score** - Harmonic mean dari precision dan recall
5. **Confusion Matrix** - Visualisasi prediksi vs aktual

### Output
- Classification report: `outputs/{task}/results/classification_report.txt`
- Confusion matrix plot: `outputs/{task}/figures/confusion_matrix.png`
- Detailed metrics: `outputs/{task}/results/evaluation_metrics.json`

### Cara Menjalankan
```bash
# Huruf
python -m src.letters.evaluate --model_path outputs/letters/models/best_model.pth --config configs/letters.yaml

# Kata
python -m src.words.evaluate --config configs/words_motion.yaml
```

### Hasil Terkini (Huruf)
Accuracy 87.82% / Precision 0.9125 / Recall 0.8782 / F1-Score 0.8808 (test set, 156 sampel, 26 kelas) — dicapai setelah normalisasi wrist-relative, augmentasi 4x pada train split, dan regularisasi (dropout 0.5, weight_decay 0.0005).

---

## Tahap 5: Test Prediksi Video Huruf (`src/letters/predict_video.py`)

### Tujuan
Menguji checkpoint huruf pada satu video mentah dari dataset, lalu menampilkan dan/atau menyimpan video beranotasi yang berisi kelas hasil prediksi dan confidence.

### Alur Inference
1. Baca konfigurasi huruf, label encoder, dan checkpoint BiLSTM
2. Ekstrak landmark dari seluruh frame video dengan parameter MediaPipe pada config
3. Terapkan normalisasi serta padding/truncating yang sama persis dengan preprocessing training
4. Prediksi satu kelas dari seluruh sequence video dan hitung confidence dengan softmax
5. Putar ulang video dengan overlay landmark tangan dan skeleton tubuh tanpa koneksi wajah dari MediaPipe Pose; tekan **Q** atau **Esc** untuk keluar. Skeleton tubuh hanya untuk visualisasi, bukan input model. Teks kelas baru muncul setelah gerakan tangan terdeteksi lalu berhenti sementara.

Prediksi dilakukan untuk **satu sequence video utuh**, bukan klasifikasi independen pada setiap frame. File landmark baru tidak dibuat.

### Input
- Video mentah, misalnya `dataset/letters/raw/B/B_0001.avi`
- Model `outputs/letters/models/best_model.pth`
- Config `configs/letters.yaml`
- Label encoder `dataset/letters/processed/label_encoder.json`

### Cara Menjalankan
```bash
# Tampilkan video hasil prediksi
python -m src.letters.predict_video --video_path dataset/letters/raw/B/B_0001.avi

# Tampilkan sekaligus simpan video beranotasi
python -m src.letters.predict_video --video_path dataset/letters/raw/B/B_0001.avi --output_path outputs/letters/predictions/B_0001_prediction.mp4

# Simpan tanpa membuka window (headless/server)
python -m src.letters.predict_video --video_path dataset/letters/raw/B/B_0001.avi --output_path outputs/letters/predictions/B_0001_prediction.mp4 --no_display

# Pilih checkpoint/config secara eksplisit bila diperlukan
python -m src.letters.predict_video --video_path <path-video> --model_path outputs/letters/models/best_model.pth --config configs/letters.yaml
```

Jika `--output_path` tidak diberikan, video ditampilkan lalu terminal menanyakan `Apakah ingin disave hasil video ini? (y/n)`. Saat menjawab `y`, Enter pada pertanyaan lokasi memakai `outputs/letters/predictions/`, atau masukkan direktori/file lain. `--output_path` menyimpan langsung tanpa pertanyaan. Jika `--no_display` dipakai tanpa `--output_path`, script hanya mencetak hasil prediksi di terminal.

Deteksi selesai gerakan dapat disesuaikan dengan `--motion_threshold` (default `0.003`) dan `--pause_frames` (default `8`). Sistem harus melihat gerakan terlebih dahulu, lalu beberapa frame diam, sebelum teks prediksi muncul. Jika jeda tidak ditemukan, hasil ditampilkan pada frame terakhir.

### Output
- Terminal: kelas huruf dan confidence
- Window OpenCV: video dengan overlay prediksi (kecuali `--no_display`)
- Video hasil opsional: path dari `--output_path`, disarankan di `outputs/letters/predictions/`

> **Konsistensi model:** `landmark.max_num_hands`, `model.input_size`, `preprocessing.max_seq_length`, dan `preprocessing.normalization` pada config harus sama dengan nilai ketika checkpoint dilatih. Config huruf saat ini memakai maksimal **2 tangan** dan `input_size: 126`.

### Inference Realtime Webcam (`src/letters/predict_webcam.py`)

Webcam OpenCV dapat dipakai untuk membaca beberapa huruf secara berulang tanpa menyiapkan file video:

```bash
python -m src.letters.predict_webcam
python -m src.letters.predict_webcam --camera_index 0 --display_width 640
```

State inference realtime:

1. **WAITING** — menunggu tangan mulai bergerak
2. **RECORDING** — mengumpulkan sequence landmark tangan
3. Setelah tangan diam selama `--pause_frames`, sequence dinormalisasi/padding dan diprediksi
4. **RESULT** — kelas serta confidence ditampilkan selama `--result_frames`
5. Sistem reset otomatis ke WAITING untuk membaca huruf berikutnya

Opsi sensitivitas utama adalah `--motion_threshold` (default `0.003`), `--pause_frames` (default `8`), dan `--result_frames` (default `45`). Tekan **Q** atau **Esc** untuk keluar. Landmark Pose badan tanpa wajah hanya visualisasi; tensor model tetap berasal dari landmark tangan. Mode webcam tidak merekam video.

---

## Struktur Folder Project

```
Model/
│
├── WORKFLOW.md                  # ← File ini (dokumentasi alur kerja)
├── README.md                    # Dokumentasi umum project
├── requirements.txt             # Daftar dependensi Python
│
├── configs/
│   ├── letters.yaml              # Konfigurasi task huruf (path, hyperparameter)
│   └── words.yaml                # Konfigurasi task kata (path, hyperparameter)
│
├── dataset/
│   ├── letters/
│   │   ├── raw/                 # Video mentah huruf, per kelas: raw/{A,B,...,Z}/
│   │   ├── landmarks/           # Hasil ekstraksi (.npy), shape (frames, 42, 3) sesuai config
│   │   └── processed/           # Data siap latih (train/val/test .pt) + label_encoder.json
│   │
│   └── words/
│       ├── raw/                 # Video mentah kata, per kelas: raw/{nama_kata}/ (belum ada isinya)
│       ├── landmarks/           # Hasil ekstraksi landmark (.npy), shape (frames, 42, 3)
│       └── processed/           # Data siap latih (train/val/test .pt) + label_encoder.json
│
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── extract_landmarks.py # Ekstraksi MediaPipe untuk huruf dan kata
│   │   ├── predict_webcam_auto.py # Realtime auto-detector (Huruf & Kata sekaligus)
│   │   └── utils.py             # Fungsi utilitas bersama
│   ├── letters/
│   │   ├── auto_recorder.py     # Perekam dataset huruf
│   │   ├── normalize_letter_videos.py
│   │   ├── preprocess.py
│   │   ├── model.py
│   │   ├── dataset_loader.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict_video.py
│   │   └── predict_webcam.py
│   └── words/
│       ├── normalize_word_videos.py
│       ├── preprocess.py
│       ├── model.py
│       ├── dataset_loader.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict_video.py
│       └── predict_webcam.py
│   └── combined/                # Pipeline gabungan 36 kelas (26 huruf + 10 kata)
│       ├── preprocess.py
│       ├── dataset_loader.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict_video.py
│       └── predict_webcam.py
├── notebooks/
│   └── exploration.ipynb        # Notebook eksplorasi & visualisasi
│
└── outputs/
    ├── letters/
    │   ├── models/               # best_model.pth huruf
    │   ├── logs/                 # training_log.csv huruf
    │   ├── figures/              # training_curves.png, confusion_matrix.png huruf
    │   ├── results/              # classification_report.txt, evaluation_metrics.json huruf
    │   └── predictions/          # Video hasil prediksi (dibuat saat diperlukan)
    │
    └── words/
        ├── models/
        ├── logs/
        ├── figures/
        └── results/
```

`dataset/*/raw|landmarks|processed/` dan `outputs/*/models|logs|figures|results/` di-gitignore (terlalu besar untuk Git) — hanya struktur folder (`.gitkeep`) yang ikut ter-commit.

---

## Urutan Eksekusi

```bash
# 1. Install dependensi
pip install -r requirements.txt

# --- Task Huruf ---
python -m src.common.extract_landmarks --input_dir dataset/letters/raw --output_dir dataset/letters/landmarks --max_num_hands 2
python -m src.letters.preprocess --input_dir dataset/letters/landmarks --output_dir dataset/letters/processed --max_seq_length 90 --normalization wrist_relative --augment_factor 3
python -m src.letters.train --config configs/letters.yaml
python -m src.letters.evaluate --model_path outputs/letters/models/best_model.pth --config configs/letters.yaml

# --- Task Kata (setelah video kata ditaruh di dataset/words/raw/{nama_kata}/) ---
python -m src.common.extract_landmarks --input_dir dataset/words/raw --output_dir dataset/words/landmarks --max_num_hands 2
python -m src.words.preprocess --config configs/words_motion.yaml
python -m src.words.train --config configs/words_motion.yaml
python -m src.words.evaluate --config configs/words_motion.yaml
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

1. **Dua task independen** — huruf (`configs/letters.yaml`, `dataset/letters/`, `outputs/letters/`) dan kata (`configs/words.yaml`, `dataset/words/`, `outputs/words/`). Jangan pernah gabungkan keduanya jadi satu model/label encoder.
2. **Dataset kata belum ada** — folder `dataset/words/raw/` masih kosong (cuma `.gitkeep`). Jangan asumsikan ada data kata sebelum dicek langsung.
3. **`max_num_hands`** — jumlah tangan harus konsisten antara `extract_landmarks.py --max_num_hands`, preprocessing, checkpoint, dan `model.input_size` di config. Config huruf saat ini memakai 2 tangan (`input_size=126`); config lain dapat memakai 1 tangan (`input_size=63`).
4. **Framework: PyTorch** — Semua pemodelan dan training menggunakan PyTorch, **BUKAN TensorFlow/Keras**.
5. **MediaPipe Hands** — Gunakan `mediapipe.solutions.hands` untuk ekstraksi landmark. Saat 2 tangan, urutan slot tetap [Left, Right] berdasarkan `multi_handedness`, tangan tak terdeteksi diisi nol per frame (bukan skip video).
6. **Normalisasi default: wrist-relative** — bukan minmax/zscore. Diterapkan per-blok 21 landmark (per tangan), bukan global.
7. **Augmentasi hanya di train split** — val/test harus selalu berupa data asli, tidak pernah diaugmentasi, supaya metrik evaluasi valid.
8. **Sequence Length** — Perlu padding/truncating agar semua video memiliki panjang sequence yang sama; cek dulu distribusi panjang video sebelum menentukan `max_seq_length` (jangan asal pakai default, terutama untuk dataset kata yang belum pernah dicek).
9. **BiLSTM** — Model menggunakan `bidirectional=True` pada `nn.LSTM` PyTorch.
10. **Data Split** — Train/Val/Test disimpan dalam folder `dataset/{task}/processed/`, bukan folder terpisah. Split dilakukan berbasis index **sebelum** normalisasi/augmentasi.
11. **Reproducibility** — Selalu set random seed untuk reproducibility.
12. **Ekstraksi landmark bersifat incremental** — video yang `.npy`-nya sudah ada otomatis di-skip; pakai `--overwrite` untuk paksa ekstrak ulang semua.
