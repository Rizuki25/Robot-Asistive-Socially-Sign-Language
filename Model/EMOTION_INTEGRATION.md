# Satu kamera untuk isyarat dan ekspresi wajah

Jalankan dari PowerShell laptop, folder `Model`, setelah stream kamera robot aktif.
Hentikan program `Emotion/fusion_webcam.py` dan program isyarat lama dahulu.

Uji tampilan kedua model tanpa mengirim gerakan:

```powershell
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --no_speech --emotion
```

Sesudah tampilan dan kecepatan diuji, tambahkan `--robot_url "http://192.168.50.2:8091"`
untuk memakai bridge Halo/Baik yang sudah aktif. Token `AINEX_RESPONSE_TOKEN` harus
sama dengan robot. Ekspresi tidak memicu gerakan pada tahap integrasi ini.

`--emotion` menambahkan Haar Cascade + YOLOv8 klasifikasi wajah pada pembaca kamera
yang sama. Frame disalin sebelum skeleton/overlay digambar. BiLSTM tetap memakai
preprocessing, urutan landmark, dan konfigurasi yang sama. Mirror hanya tampilan.

YOLO menerima crop wajah terbesar dengan padding 20%, ukuran inferensi 224,
dan nama kelas dari checkpoint. Confidence di bawah 0.40 ditampilkan sebagai
`Belum yakin`; selisih dua kelas teratas di bawah 0.20 menghasilkan label majemuk,
sesuai aturan visual program Emotion. Ini label ekspresi prediksi model.

Default model: `Emotion/webcam/models/fer2013_baseline-2/weights/best.pt`,
di-resolve terhadap lokasi source, bukan current working directory.
Override dengan `--emotion_model PATH` (path relatif mengikuti working directory).

Worker wajah memakai CPU (`--emotion_device cpu`) dengan interval minimum
0.2 detik (`--emotion_interval 0.2`), paling banyak satu frame tertunda yang
selalu diganti oleh frame terbaru. Ini tidak menjamin FPS tertentu karena CPU
tetap dipakai bersama MediaPipe. Hasil lebih tua dari 1.5 detik sejak frame
diserahkan tidak ditampilkan. Tidak ada wajah berarti label lama dihapus.
Selama respons robot, kedua klasifikasi dijeda dan hasil wajah dibersihkan.

`--no_speech` pada perintah ini mematikan audio keluaran isyarat di laptop.
Integrasi ini tidak memuat model audio emosi atau mikrofon, tidak mengirim
heartbeat expression untuk pengumuman kesiapan.
Tanpa `--emotion`, Ultralytics tidak diimpor oleh program isyarat.

## Hasil di aplikasi web

Setelah backend `app-mobile` aktif di port 3001, tambahkan `--web_url`:

```powershell
python -u -m src.combined.predict_webcam --config configs/combined_90.yaml --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --no_speech --emotion --web_url http://localhost:3001 --web_room demo-ta
```

Hasil isyarat terkunci dikirim ke `/api/sign-result`. Status ekspresi dikirim
paling banyak dua kali per detik ke `/api/emotion-result`, memakai worker jaringan
terpisah. Jika backend membutuhkan API key, set `MODEL_API_KEY` yang sama di
PowerShell model. Model tidak perlu memakai URL tunnel jika backend ada di laptop
yang sama. Ini tidak mengaktifkan gerakan robot; gerakan tetap memerlukan `--robot_url`.

Frontend menampilkan animasi wajah dan teks pada Bahasa Isyarat dan Mode Percakapan.
Hasil emosi tidak menjadi pesan chat dan tidak memicu text-to-speech. Wajah hilang,
hasil belum yakin, jeda, dan error dikirim sebagai status tanpa label emosi lama.
Jika pembaruan berhenti, panel kedaluwarsa setelah 4 detik. Saat HTTP gagal, pesan
isyarat tidak dicoba ulang untuk menghindari duplikasi chat/suara.

Pemeriksaan software dari folder Model:

```powershell
python -m unittest discover -s tests -p test_emotion_recognition.py -v
```

Tes ini memeriksa pemetaan kelas, ketidakpastian, hasil kedaluwarsa, wajah hilang,
frame terbaru, serta pembatalan hasil saat jeda. Tes tidak membuktikan akurasi
model atau latensi kamera-ke-hasil; keduanya perlu pengujian pada stream robot.
