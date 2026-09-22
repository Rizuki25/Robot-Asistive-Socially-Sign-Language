# Pengujian latensi kamera robot (Tabel 3.4)

Program `src.combined.predict_webcam` menampilkan **Latensi terakhir** (ms)
di bawah header kamera dan mencetak `[LATENSI]` di terminal setiap hasil terkunci.
Pengukuran berlaku untuk webcam lokal maupun stream kamera robot.

Jalankan dari folder `Model` untuk merekam percobaan kelas Halo:

```powershell
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --no_speech --emotion --latency_csv outputs/combined_90/latency_halo_01.csv --latency_label Halo
```

Lakukan gestur Halo berulang kali, mengikuti status siap pada layar, lalu tekan
Q/Esc. Setiap hasil terkunci menghasilkan satu baris CSV. Untuk Apa, Kamu, dan
Saya, jalankan sesi baru dengan mengganti `--latency_label` dan nama file CSV.
Gunakan nama file baru setiap sesi; file lama tidak ditimpa. Lokasi CSV relatif
terhadap direktori terminal. Opsi aplikasi mobile dan respons robot yang biasa
digunakan dapat tetap ditambahkan pada command tersebut.

| Kolom CSV | Isi untuk tabel pengujian |
| --- | --- |
| no | Nomor hasil dalam sesi |
| kelas_gestur | Kelas aktual yang ditetapkan melalui `--latency_label` |
| prediksi | Kelas keluaran model |
| waktu_mulai_deteksi_ms | Waktu gerakan pemicu terdeteksi, relatif terhadap awal sesi pengukuran |
| waktu_prediksi_ms | Waktu hasil prediksi terkunci, relatif terhadap awal sesi yang sama |
| latensi_ms | Waktu prediksi dikurangi waktu mulai deteksi |
| status | Sesuai / Tidak sesuai terhadap kelas aktual; Belum dinilai jika kelas aktual tidak diberikan |
| confidence | Confidence hasil pada skala 0–1 |
| sumber | Voting stabil atau fallback jeda |

`--latency_label` hanya memberi label aktual untuk evaluasi, tidak mengubah
prediksi model. Status menilai kecocokan kelas, bukan kelulusan ambang latensi.
Tanpa `--latency_csv`, angka tetap terlihat di layar dan terminal tanpa disimpan.

## Batas pengukuran untuk laporan

Jam pengukuran menggunakan `time.perf_counter()` yang monotonik. Titik mulai
adalah saat gerakan pemicu pertama terdeteksi oleh pipeline di laptop. Untuk
gestur berikutnya, titik mulai adalah frame pertama dari rangkaian gerak yang
berhasil memenuhi syarat rearm; rangkaian yang terputus dibatalkan.
Titik akhir adalah saat hasil dikunci oleh voting atau fallback, sebelum
pemutaran audio, pengiriman ke aplikasi, dan perintah robot.

Latensi mencakup pengumpulan frame, pemrosesan lanjutan, inferensi dan voting,
termasuk waktu menunggu frame berikutnya. Ini **bukan waktu inferensi model
saja**, bukan waktu sejak pengguna selesai melakukan gestur, dan bukan latensi
end-to-end dari sensor kamera robot. Waktu capture/transmisi gambar pemicu
sebelum terdeteksi di laptop tidak diukur karena stream tidak menyediakan
timestamp sensor tersinkronisasi untuk pengukuran ini. Rendering layar,
pengiriman ke aplikasi, dan durasi respons robot juga tidak termasuk.

Hanya percobaan yang menghasilkan prediksi terkunci dicatat. Gestur yang tidak
memicu deteksi atau sesi yang dihentikan sebelum hasil muncul perlu dicatat
manual sebagai percobaan tanpa hasil, jangan dianggap latensi nol. Gunakan
kondisi daya, koneksi kamera, dan opsi emosi yang konsisten antar percobaan.
