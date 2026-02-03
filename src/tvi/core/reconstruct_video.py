import numpy as np

from tvi.utils.time_grid import build_time_grids
from tvi.utils.window import centered_window_indices
from tvi.utils.rescale import rescale_to_local
from tvi.interpolation.neville import neville_eval


def reconstruct_video_gray_neville(
    frames_full: np.ndarray,
    stride: int,
    m: int = 8,
    use_chebyshev: bool = False,
    M_avail: int = 16,
) -> np.ndarray:
    """
    Reconstruct all intermediate frames from subsampled frames using local polynomial interpolation in time.

    Parameters
    ----------
    frames_full : (T, H, W) float/uint8
        Original full frames (used here only to generate sampled subset and to keep sampled frames exact).
        In a real use-case you could instead pass only sampled frames; but this makes testing simple.
    stride : int
        Keep every stride-th frame as samples.
    m : int
        Local interpolation window size (polynomial degree <= m-1).
    use_chebyshev : bool
        If True, selects m nodes from a larger candidate window size M_avail with Chebyshev-like distribution.
        Requires tvi.utils.chebyshev_select.select_chebyshev_indices.
    M_avail : int
        Candidate window size for Chebyshev selection (must be >= m).

    Returns
    -------
    frames_recon : (T, H, W) float64
    """
    if frames.ndim not in (3, 4):
        raise ValueError("frames must be (T,H,W) bw or (T,H,W,3) rgb")

    T = frames.shape[0]
    if frames.ndim == 3:
        H, W = frames.shape[1], frames.shape[2]
        C = 1
    else:
        H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]


    T, H, W = frames_full.shape
    t_full, t_sampled, t_recon = build_time_grids(num_frames=T, stride=stride)

    # sampled frames (K,H,W)
    frames_sampled = frames_full[t_sampled].astype(np.float64)
    K = len(t_sampled)

    if m > K:
        raise ValueError(f"m={m} cannot exceed number of sampled frames K={K}")

    # mask sampled times for exact copy
    is_sampled = np.zeros(T, dtype=bool)
    is_sampled[t_sampled] = True

    # flatten sampled frames to (K, P)
    sampled = frames[t_sampled].astype(np.float64)
    K = sampled.shape[0]
    P = H * W * C

    Y = sampled.reshape(K, P)
    out = np.empty((T, P), dtype=np.float64)
    

    # optional chebyshev selector
    if use_chebyshev:
        from tvi.utils.chebyshev_select import select_chebyshev_indices
        if M_avail < m:
            raise ValueError("M_avail must be >= m when use_chebyshev=True")
        if M_avail > K:
            M_avail = K  # clamp

    for t in t_recon:
        
        if t % 50 == 0:
            print(f"reconstructing frame {t}/{T}")

        if is_sampled[t]:
            # exact sampled frame
            out[t] = frames_full[t].reshape(P).astype(np.float64)
            continue

        # nearest sampled index (sampled-index space)
        k_center = int(np.argmin(np.abs(t_sampled - t)))

        if use_chebyshev:
            sel = select_chebyshev_indices(k_total=K, k_center=k_center, m=m, M_avail=M_avail)
        else:
            sel = centered_window_indices(k_total=K, k_center=k_center, m=m)

        x_nodes = t_sampled[sel].astype(np.float64)      # (m,)
        y_nodes = Y[sel]                                 # (m,P)

        # rescale time
        xi, xi_nodes = rescale_to_local(float(t), x_nodes, h=float(stride))

        # Neville eval gives (P,)
        out[t] = neville_eval(xi, xi_nodes, y_nodes)
        out[t] = frames[t].reshape(P).astype(np.float64)

    if C == 1:
        return out.reshape(T, H, W)
    return out.reshape(T, H, W, C)


