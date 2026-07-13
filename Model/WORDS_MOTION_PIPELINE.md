# Pipeline Khusus Kata (Motion-Aware)

Pipeline ini terpisah dari pipeline huruf. Ia tidak membaca atau menulis
`dataset/letters`, `outputs/letters`, maupun `configs/letters.yaml`.

## Fitur per frame

Setiap frame menghasilkan 140 fitur:

- 126 fitur bentuk dua tangan yang dinormalisasi relatif terhadap wrist;
- 6 koordinat posisi global wrist kiri dan kanan;
- 6 koordinat perpindahan wrist dari posisi valid pertama;
- 2 mask kehadiran tangan.

Data juga menyimpan panjang sequence asli. `WordMotionBiLSTM` menggunakan
`pack_padded_sequence`, sehingga frame padding tidak dibaca sebagai gerakan.

## Baseline tanpa augmentasi

Jalankan dari folder `Model`:

```powershell
python src/preprocess_words.py --config configs/words_motion.yaml
python src/train_words.py --config configs/words_motion.yaml
python src/evaluate_words.py --config configs/words_motion.yaml
```

Output baseline tersimpan terpisah:

```text
dataset/words/processed_motion/
outputs/words_motion/
```

Konfigurasi awal memakai `max_seq_length: 100` untuk dataset lama yang memiliki
P95 98 frame. Untuk dataset baru 3 detik dan 30 FPS, periksa jumlah frame hasil
ekstraksi lalu ubah menjadi 90 jika seluruh video benar-benar sekitar 90 frame.

## Menambah augmentasi secara bertahap

Baseline wajib dijalankan dahulu dengan `augment_factor: 0`. Jika train accuracy
meningkat dengan sehat dan validation tidak mengalami class collapse, lakukan
eksperimen berikut secara satu per satu:

```powershell
python src/preprocess_words.py --config configs/words_motion.yaml --augment_factor 1
python src/train_words.py --config configs/words_motion.yaml
python src/evaluate_words.py --config configs/words_motion.yaml
```

Jika factor 1 membantu, lanjutkan ke factor 2. Jangan langsung memakai factor 3.
Setiap preprocessing/training baru menimpa hasil `words_motion` sebelumnya, jadi
salin output eksperimen yang ingin dipertahankan sebelum menjalankan eksperimen
berikutnya.
