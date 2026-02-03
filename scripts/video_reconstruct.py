import argparse
import os
import cv2
import numpy as np

from tvi.utils.time_grid import build_time_grids
from tvi.utils.window import centered_window_indices
from tvi.utils.rescale import rescale_to_local
from tvi.interpolation.neville import neville_eval
from tvi.interpolation.barycentric import barycentric_weights, barycentric_eval


def read_video(path: str, rgb: bool, max_frames: int = 0) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    collected_frames: list[np.ndarray] = []
    n_read = 0

    while True:
        # stop early if requested
        if max_frames > 0 and n_read >= max_frames:
            break

        ok, frame_bgr = cap.read()
        if not ok:
            break

        if rgb:
            collected_frames.append(frame_bgr.astype(np.float64))  # (H,W,3) BGR
        else:
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            collected_frames.append(frame_gray.astype(np.float64))  # (H,W)

        n_read += 1

    cap.release()

    if len(collected_frames) == 0:
        raise RuntimeError(f"No frames read from: {path}")

    return np.stack(collected_frames, axis=0), fps


def write_video(path: str, frames: np.ndarray, fps: float, rgb: bool) -> None:
    """
    Write frames to mp4.
    - bw: frames (T,H,W) -> converted to BGR for writing
    - rgb: frames (T,H,W,3) assumed BGR
    """
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


def reconstruct_video(
    frames: np.ndarray,
    stride: int,
    m: int,
    method: str,
    use_rescale: bool,
    bias: int,
    use_chebyshev: bool,
    M_avail: int,
    progress_every: int,
) -> np.ndarray:
    """
    frames: (T,H,W) bw or (T,H,W,3) rgb, float64 in [0,255]
    returns: reconstructed frames with same shape
    """
    if frames.ndim not in (3, 4):
        raise ValueError("frames must be (T,H,W) for bw or (T,H,W,3) for rgb")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if m <= 0:
        raise ValueError("m must be positive")

    T = frames.shape[0]
    if frames.ndim == 3:
        H, W = frames.shape[1], frames.shape[2]
        C = 1
    else:
        H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]

    t_full, t_sampled, t_recon = build_time_grids(num_frames=T, stride=stride)
    K = len(t_sampled)
    if m > K:
        raise ValueError(f"m={m} > K={K} sampled frames. Reduce m or reduce stride.")

    sampled = frames[t_sampled].astype(np.float64)   # (K,H,W) or (K,H,W,3)
    P = H * W * C
    Y = sampled.reshape(K, P)                        # (K,P)

    is_sampled = np.zeros(T, dtype=bool)
    is_sampled[t_sampled] = True

    if use_chebyshev:
        from tvi.utils.chebyshev_select import select_chebyshev_indices
        if M_avail < m:
            raise ValueError("M_avail must be >= m when using Chebyshev")
        if M_avail > K:
            M_avail = K

    out = np.empty((T, P), dtype=np.float64)

    for t in t_recon:
        if progress_every > 0 and (t % progress_every == 0):
            print(f"[reconstruct] frame {t}/{T}")

        if is_sampled[t]:
            out[t] = frames[t].reshape(P).astype(np.float64)
            continue

        k_center = int(np.argmin(np.abs(t_sampled - t)))
        k_center = int(np.clip(k_center + bias, 0, K - 1))

        if use_chebyshev:
            sel = select_chebyshev_indices(k_total=K, k_center=k_center, m=m, M_avail=M_avail)
        else:
            sel = centered_window_indices(k_total=K, k_center=k_center, m=m)

        x_nodes = t_sampled[sel].astype(np.float64)  # (m,)
        y_nodes = Y[sel]                              # (m,P)

        if method == "neville":
            if use_rescale:
                xi, xi_nodes = rescale_to_local(float(t), x_nodes, h=float(stride))
                out[t] = neville_eval(xi, xi_nodes, y_nodes)
            else:
                out[t] = neville_eval(float(t), x_nodes, y_nodes)

        elif method == "barycentric":
            w = barycentric_weights(x_nodes)
            out[t] = barycentric_eval(float(t), x_nodes, y_nodes, w)
        else:
            raise ValueError("method must be 'neville' or 'barycentric'")

    if C == 1:
        return out.reshape(T, H, W)
    return out.reshape(T, H, W, C)


def main():
    parser = argparse.ArgumentParser(
        description="Temporal video interpolation via local polynomial methods"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/out.mp4")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bw", action="store_true", help="process grayscale (default)")
    mode.add_argument("--rgb", action="store_true", help="process color (BGR)")

    parser.add_argument("--stride", type=int, default=5, help="keep every stride-th frame")
    parser.add_argument("--m", type=int, default=8, help="local window size (degree <= m-1)")
    parser.add_argument("--method", type=str, default="barycentric", choices=["neville", "barycentric"])
    parser.add_argument("--no_rescale", action="store_true", help="disable time rescaling (more chaotic)")
    parser.add_argument("--bias", type=int, default=0, help="bias window center in sampled-index space")

    parser.add_argument("--chebyshev", action="store_true", help="select nodes via Chebyshev-like distribution")
    parser.add_argument("--M_avail", type=int, default=16, help="candidate window size for Chebyshev selection")

    parser.add_argument("--frames", type=int, default=0, help="process only first N frames (0 = all)")
    parser.add_argument("--resize", type=float, default=1.0, help="scale factor for H,W (e.g. 0.5 for half res)")

    parser.add_argument("--progress_every", type=int, default=50, help="print progress every N frames")
    parser.add_argument("--crazy", action="store_true", help="apply a preset for strong artifacts")

    args = parser.parse_args()
    is_rgb = bool(args.rgb)  # default False => bw

    # Preset
    if args.crazy:
        args.stride = max(args.stride, 12)
        args.m = max(args.m, 16)
        args.bias = args.bias if args.bias != 0 else 2
        args.no_rescale = True

    print("[stage] reading video...")
    frames, fps = read_video(args.input, rgb=is_rgb, max_frames=args.frames)




    # Resize (both bw and rgb)
    if args.resize != 1.0:
        scale = float(args.resize)
        if not (0.1 <= scale <= 2.0):
            raise ValueError("resize should be between 0.1 and 2.0")

        if is_rgb:
            T, H, W, C = frames.shape
            newH = max(1, int(round(H * scale)))
            newW = max(1, int(round(W * scale)))
            resized = np.empty((T, newH, newW, C), dtype=np.float64)
            for t in range(T):
                resized[t] = cv2.resize(frames[t], (newW, newH), interpolation=cv2.INTER_AREA)
            frames = resized
        else:
            T, H, W = frames.shape
            newH = max(1, int(round(H * scale)))
            newW = max(1, int(round(W * scale)))
            resized = np.empty((T, newH, newW), dtype=np.float64)
            for t in range(T):
                resized[t] = cv2.resize(frames[t], (newW, newH), interpolation=cv2.INTER_AREA)
            frames = resized

    print("Loaded:", args.input, "frames:", frames.shape, "fps:", fps)
    print("Mode:", "rgb" if is_rgb else "bw")
    print("Settings:",
          "stride=", args.stride,
          "m=", args.m,
          "method=", args.method,
          "rescale=", (not args.no_rescale),
          "bias=", args.bias,
          "chebyshev=", args.chebyshev,
          "M_avail=", args.M_avail)

    recon = reconstruct_video(
        frames=frames,
        stride=args.stride,
        m=args.m,
        method=args.method,
        use_rescale=(not args.no_rescale),
        bias=args.bias,
        use_chebyshev=args.chebyshev,
        M_avail=args.M_avail,
        progress_every=args.progress_every,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_video(args.output, recon, fps, rgb=is_rgb)
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
