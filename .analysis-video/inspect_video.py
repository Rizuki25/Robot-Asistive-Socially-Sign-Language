from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent
cap = cv2.VideoCapture(str(root / 'Video Project 3 (1).mp4'))
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print({'fps': fps, 'frames': count, 'duration': count / fps,
       'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})
tiles = []
for second in range(0, int(count / fps) + 1, 2):
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = cap.read()
    if not ok:
        continue
    cv2.imwrite(str(out / f'frame_{second:03d}.jpg'), frame)
    h, w = frame.shape[:2]
    tile = cv2.resize(frame, (640, round(h * 640 / w)))
    tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(tile, f'{second}s', (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
    tiles.append(tile)
for start in range(0, len(tiles), 8):
    batch = tiles[start:start+8]
    while len(batch) % 2:
        batch.append(np.zeros_like(batch[0]))
    sheet = np.vstack([np.hstack(batch[i:i+2]) for i in range(0,len(batch),2)])
    dest = out / f'sheet_{start//8}.jpg'
    cv2.imwrite(str(dest), sheet)
    print(dest)
tiles = []
for index in range(19):
    second = index * 0.5
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = cap.read()
    if not ok:
        continue
    crop = frame[130:960, 420:1460]
    tile = cv2.resize(crop, (520,415))
    tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(tile, f'{second:.1f}s', (10,21), cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    tiles.append(tile)
for start in range(0, len(tiles), 6):
    batch = tiles[start:start+6]
    while len(batch) % 2:
        batch.append(np.zeros_like(batch[0]))
    cv2.imwrite(str(out / f'detail_{start//6}.jpg'), np.vstack([np.hstack(batch[i:i+2]) for i in range(0,len(batch),2)]))
cap.release()
