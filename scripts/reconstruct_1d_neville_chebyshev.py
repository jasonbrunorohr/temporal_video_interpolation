import numpy as np

from tvi.utils.time_grid import build_time_grids
from tvi.utils.chebyshev_select import select_chebyshev_indices
from tvi.utils.rescale import rescale_to_local
from tvi.interpolation.neville import neville_eval


def main():
    num_frames = 100
    stride = 5

    # Interpolation degree knob
    m = 4         # try 4 then 8
    M_avail = 16 # candidate window size (must be >= m)

    t_full, t_sampled, t_recon = build_time_grids(num_frames=num_frames, stride=stride)

    # Choose one:
    # (A) lower frequency (clean comparison)
    # f_full = np.sin(0.2 * t_full) + 1.0
    # (B) higher frequency (more chaotic)
    f_full = np.sin(0.2*t_full) + 1 + (t_full >= 50).astype(float)


    # OPTIONAL: add step for Gibbs-like artifacts
    # f_full = f_full + (t_full >= 50).astype(float)

    f_sampled = f_full[t_sampled]

    is_sampled = np.zeros_like(f_full, dtype=bool)
    is_sampled[t_sampled] = True

    f_recon = np.empty_like(f_full)

    for t in t_recon:
        if is_sampled[t]:
            f_recon[t] = f_full[t]
        else:
            # nearest sampled index (in sampled-index space)
            k_center = int(np.argmin(np.abs(t_sampled - t)))

            # SELECT Chebyshev-like nodes from a larger candidate window
            sel = select_chebyshev_indices(
                k_total=len(t_sampled),
                k_center=k_center,
                m=m,
                M_avail=M_avail
            )

            x_nodes = t_sampled[sel]              # (m,)
            y_nodes = f_sampled[sel][:, None]     # (m, 1)

            # rescale to local coordinate (conditioning)
            xi, xi_nodes = rescale_to_local(float(t), x_nodes, h=float(stride))

            val = neville_eval(xi, xi_nodes, y_nodes)  # (1,)
            f_recon[t] = val[0]

    # Diagnostics
    missing = ~is_sampled
    max_missing_error = np.max(np.abs(f_recon[missing] - f_full[missing]))
    max_sampled_error = np.max(np.abs(f_recon[is_sampled] - f_full[is_sampled]))

    print("m =", m, "M_avail =", M_avail)
    print("max sampled error:", max_sampled_error)
    print("max missing error:", max_missing_error)

    # Small table (change the range depending on what you study)
    for t in range(0, 23):
        tag = "S" if is_sampled[t] else "M"
        err = abs(f_recon[t] - f_full[t])
        print(t, tag, f_full[t], f_recon[t], err)


if __name__ == "__main__":
    main()
