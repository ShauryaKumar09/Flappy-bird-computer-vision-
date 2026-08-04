"""Phase 0: prove camera capture + PoseLandmarker inference actually work end to end."""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "pose_landmarker_lite.task"

landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
)
print("landmarker created OK")

# Inference on a synthetic frame proves the graph runs even with no camera access.
blank = mp.Image(
    image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), dtype=np.uint8)
)
res = landmarker.detect_for_video(blank, 0)
print(f"synthetic frame -> {len(res.pose_landmarks)} poses (0 expected)")

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("CAMERA: could not open index 0 (check macOS camera permission for this app)")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detected = 0
times = []
t_end = time.monotonic() + 6.0
n = 0
while time.monotonic() < t_end:
    ok, frame = cap.read()
    if not ok:
        continue
    n += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    t0 = time.perf_counter()
    res = landmarker.detect_for_video(img, int(time.monotonic() * 1000))
    times.append((time.perf_counter() - t0) * 1000)
    if res.pose_landmarks:
        detected += 1
        if detected == 1:
            lm = res.pose_landmarks[0]
            ls, rs = lm[11], lm[12]
            sw = ((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) ** 0.5
            print(f"  first pose: {len(lm)} landmarks, shoulder width = {sw:.3f}")

cap.release()
print(f"frames read      : {n}")
print(f"frames with pose : {detected}")
if times:
    print(f"inference ms     : mean {np.mean(times):.1f}  p95 {np.percentile(times, 95):.1f}")
