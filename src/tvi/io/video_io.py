import cv2
import numpy as np

def read_video(path: str, rgb: bool) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rgb:
            # OpenCV reads BGR; keep as BGR for writing (no conversion cost)
            frames.append(frame.astype(np.float64))      # (H,W,3)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float64))       # (H,W)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from: {path}")

    return np.stack(frames, axis=0), fps


def write_video(path: str, frames: np.ndarray, fps: float, rgb: bool) -> None:
    if rgb:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("RGB mode requires frames shape (T,H,W,3)")
        T, H, W, _ = frames.shape
    else:
        if frames.ndim != 3:
            raise ValueError("BW mode requires frames shape (T,H,W)")
        T, H, W = frames.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (W, H), isColor=True)
    if not out.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {path}")

    frames_u8 = np.clip(frames, 0, 255).astype(np.uint8)

    for t in range(T):
        if rgb:
            out.write(frames_u8[t])  # already BGR
        else:
            bgr = cv2.cvtColor(frames_u8[t], cv2.COLOR_GRAY2BGR)
            out.write(bgr)

    out.release()
