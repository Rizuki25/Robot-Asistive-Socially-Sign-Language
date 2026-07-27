import os
import time

import cv2


TARGET_FPS = 30.0
TARGET_FRAMES = 90
MIN_CAPTURE_FPS = 27.0
COUNTDOWN_DURATION = 2.0

DATASET_ROOT = r"D:\dataset"


def verify_video(video_path):
    """Dekode ulang video dan kembalikan jumlah frame serta FPS tersimpan."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0

    stored_fps = float(cap.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        decoded_frames += 1
    cap.release()
    return decoded_frames, stored_fps


def save_verified_video(frames, filename, fourcc, frame_size):
    """Simpan tepat TARGET_FRAMES dan masukkan ke dataset setelah lolos verifikasi."""
    if len(frames) != TARGET_FRAMES:
        print(
            f"[GAGAL] Jumlah frame tangkapan {len(frames)}, "
            f"seharusnya {TARGET_FRAMES}."
        )
        return False

    base, extension = os.path.splitext(filename)
    temporary_filename = f"{base}.tmp{extension}"

    writer = cv2.VideoWriter(
        temporary_filename,
        fourcc,
        TARGET_FPS,
        frame_size,
    )
    if not writer.isOpened():
        print(f"[GAGAL] VideoWriter tidak dapat dibuat: {temporary_filename}")
        return False

    for frame in frames:
        resized = cv2.resize(frame, frame_size)
        writer.write(resized)
    writer.release()

    decoded_frames, stored_fps = verify_video(temporary_filename)
    fps_is_valid = abs(stored_fps - TARGET_FPS) <= 0.1
    if decoded_frames != TARGET_FRAMES or not fps_is_valid:
        print(
            f"[GAGAL] Verifikasi video: {decoded_frames} frame, "
            f"{stored_fps:.3f} FPS. File tidak dimasukkan ke dataset."
        )
        if os.path.exists(temporary_filename):
            os.remove(temporary_filename)
        return False

    os.replace(temporary_filename, filename)
    print(
        f"[OK] Video tersimpan dan terverifikasi: {filename} | "
        f"{decoded_frames} frame @ {stored_fps:.1f} FPS"
    )
    return True


def get_camera_name(index):
    """Coba dapatkan nama kamera menggunakan DirectShow backend."""
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            backend = cap.getBackendName()
            cap.release()
            return backend
        cap.release()
    except:
        pass
    return "Unknown"


def detect_all_cameras(max_index=10):
    """Scan semua index kamera dan kembalikan daftar kamera yang tersedia."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # Ambil resolusi asli sebagai info tambahan
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend = cap.getBackendName()
                available.append(
                    {"index": i, "width": w, "height": h, "backend": backend}
                )
            cap.release()
        else:
            cap.release()
    return available


def is_likely_external(cam_info):
    """Heuristik: kamera di index > 0 kemungkinan besar kamera eksternal."""
    return cam_info["index"] > 0


def main():
    print("=" * 50)
    print("  DETEKSI KAMERA")
    print("=" * 50)
    print("Scanning semua kamera yang tersedia...\n")

    cameras = detect_all_cameras()

    if not cameras:
        print("Tidak ada kamera yang dapat diakses.")
        return

    # Tampilkan semua kamera yang ditemukan
    print(f"Ditemukan {len(cameras)} kamera:\n")
    for idx, cam in enumerate(cameras):
        label = (
            "EKSTERNAL (kemungkinan)"
            if is_likely_external(cam)
            else "BUILT-IN (kemungkinan)"
        )
        print(
            f"  [{idx}] Index {cam['index']} - {cam['backend']} - {cam['width']}x{cam['height']} - {label}"
        )

    print()

    # Pilih kamera: prioritaskan kamera eksternal
    external_cams = [c for c in cameras if is_likely_external(c)]

    if len(cameras) == 1:
        selected = cameras[0]
        print(f"Hanya 1 kamera ditemukan, menggunakan index {selected['index']}.")
    elif external_cams:
        # Ada kamera eksternal, tapi beri pilihan ke user
        print(f"Kamera eksternal terdeteksi di index {external_cams[0]['index']}.")
        choice = input(
            f"Gunakan kamera eksternal? (Y/n, atau ketik nomor [0-{len(cameras) - 1}]): "
        ).strip()

        if choice.lower() == "n":
            selected = cameras[0]  # fallback ke kamera pertama (biasanya built-in)
        elif choice.isdigit() and 0 <= int(choice) < len(cameras):
            selected = cameras[int(choice)]
        else:
            selected = external_cams[0]  # default: kamera eksternal pertama
    else:
        selected = cameras[0]
        print(
            f"Tidak ada kamera eksternal terdeteksi, menggunakan index {selected['index']}."
        )

    print(
        f"\n>> Menggunakan kamera index {selected['index']} ({selected['width']}x{selected['height']})"
    )

    # Buka kamera yang dipilih dengan DirectShow
    cap = cv2.VideoCapture(selected["index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Gagal membuka kamera yang dipilih.")
        return

    # Minta kamera mengirim 30 FPS. Nilai ini tetap diverifikasi dari laju
    # tangkapan aktual karena tidak semua kamera mematuhi cap.set().
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(
        f"Target rekaman: {TARGET_FRAMES} frame @ {TARGET_FPS:.0f} FPS. "
        f"Kamera melaporkan {reported_fps:.2f} FPS."
    )

    # Biarkan kamera merekam pada resolusi aslinya
    # Kita tidak perlu lagi memaksa cap.set(...) melainkan meresizenya saat menyimpan

    # Konfigurasi Resolusi Video yang Disimpan
    # Menggunakan rasio 4:3 agar video terlihat lebih kotak dan tidak gepeng
    out_width = 1440
    out_height = 1080

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    current_filename = ""

    state = "WAITING"  # Pilihan: WAITING, COUNTDOWN, RECORDING
    start_time = 0
    capture_start_time = 0
    recorded_frames = []

    current_alphabet = "A"  # Folder dan label huruf default

    print("Kamera aktif. Tekan tombol A-Z untuk memilih folder/label.")
    print("Tekan SPACE (Spasi) untuk mulai merekam, atau ESC untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.perf_counter()
        display_frame = frame.copy()

        # Tampilkan alfabet yang dipilih pada ujung kanan atas
        cv2.putText(
            display_frame,
            f"Folder/Label: {current_alphabet}",
            (frame.shape[1] - 350, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

        if state == "WAITING":
            cv2.putText(
                display_frame,
                "Tekan 'SPACE' untuk Merekam",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        elif state == "COUNTDOWN":
            elapsed = current_time - start_time
            remaining = COUNTDOWN_DURATION - elapsed
            if remaining <= 0:
                state = "RECORDING"

                # Buat folder khusus per abjad
                save_dir = os.path.join(DATASET_ROOT, current_alphabet)
                os.makedirs(save_dir, exist_ok=True)

                # Buat nama file baru berdasarkan waktu saat ini di dalam folder alfabet tersebut
                current_filename = os.path.join(
                    save_dir,
                    f"video_{current_alphabet}_{time.strftime('%Y%m%d_%H%M%S')}.avi",
                )
                recorded_frames = []
                capture_start_time = time.perf_counter()
            else:
                cv2.putText(
                    display_frame,
                    f"Mulai dalam: {int(remaining) + 1}",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 165, 255),
                    3,
                )
        elif state == "RECORDING":
            # Setiap iterasi di blok ini berasal dari satu cap.read() yang sukses.
            # Berhenti berdasarkan jumlah frame, bukan berdasarkan timer.
            recorded_frames.append(frame.copy())
            elapsed = max(time.perf_counter() - capture_start_time, 1e-6)
            capture_fps = len(recorded_frames) / elapsed

            cv2.putText(
                display_frame,
                "MEREKAM...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"Frame: {len(recorded_frames)}/{TARGET_FRAMES} | "
                f"Capture: {capture_fps:.1f} FPS",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            if len(recorded_frames) >= TARGET_FRAMES:
                state = "WAITING"
                actual_duration = max(
                    time.perf_counter() - capture_start_time,
                    1e-6,
                )
                actual_capture_fps = len(recorded_frames) / actual_duration

                if actual_capture_fps < MIN_CAPTURE_FPS:
                    print(
                        f"[DITOLAK] Kamera hanya menangkap "
                        f"{actual_capture_fps:.2f} FPS "
                        f"(minimum {MIN_CAPTURE_FPS:.1f} FPS). "
                        "Periksa pencahayaan/resolusi lalu rekam ulang."
                    )
                else:
                    saved = save_verified_video(
                        recorded_frames,
                        current_filename,
                        fourcc,
                        (out_width, out_height),
                    )
                    if saved:
                        print(
                            f"Capture aktual: {actual_capture_fps:.2f} FPS "
                            f"selama {actual_duration:.2f} detik."
                        )

                recorded_frames = []

        cv2.imshow("Kamera Perekam (Logitech)", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Keluar ketika tombol ESC (27) ditekan
        if key == 27:
            break

        # Merekam ketika SPACE ditekan
        elif key == 32 and state == "WAITING":
            state = "COUNTDOWN"
            start_time = time.perf_counter()

        # Memilih alfabet A-Z (huruf kecil: 97-122, huruf kapital: 65-90)
        elif 97 <= key <= 122:
            current_alphabet = chr(key).upper()
        elif 65 <= key <= 90:
            current_alphabet = chr(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
