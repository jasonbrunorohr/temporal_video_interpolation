import cv2
import numpy as np


def load_video_grayscale(path: str) -> np.ndarray:
    """
    Load a video and return a 3D array:
        frames[t] ∈ R^{H×W}
    """
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))

    cap.release()
    return np.stack(frames, axis=0)


def temporal_subsample(frames: np.ndarray, stride: int) -> np.ndarray:
    """
    Keep every `stride`-th frame.

    frames: array of shape (T, H, W)
    returns: array of shape (⌊T/stride⌋, H, W)
    """
    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    return frames[::stride]


if __name__ == "__main__":
    video_path = "data/Gaga.mp4"
    stride = 5

    frames = load_video_grayscale(video_path)
    subsampled = temporal_subsample(frames, stride)

    print("Original shape (T, H, W):", frames.shape)
    print("Subsampled shape:", subsampled.shape)
    print("Stride:", stride)
