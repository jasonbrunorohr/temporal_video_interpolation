import numpy as np

from tvi.utils.time_grid import build_time_grids
from tvi.utils.window import centered_window_indices
from tvi.utils.rescale import rescale_to_local
from tvi.interpolation.neville import neville_eval


def main():
    num_frames = 100
    stride = 5
    m = 4  # try 4, then 8

    t_full, t_sampled, t_recon = build_time_grids(num_frames=num_frames, stride=stride)

    # Smooth signal first (then you can add a step later)
    f_full = np.sin(0.2 * t_full) + 1.0

    f_sampled = f_full[t_sampled]

    is_sampled = np.zeros_like(f_full, dtype=bool)
    is_sampled[t_sampled] = True

    f_recon = np.empty_like(f_full)

    for t in t_recon:
        if is_sampled[t]:
            f_recon[t] = f_full[t]
        else:
            k_center = int(np.argmin(np.abs(t_sampled - t)))
            window = centered_window_indices(k_total=len(t_sampled), k_center=k_center, m=m)

            x_nodes = t_sampled[window]              # (m,)
            y_nodes = f_sampled[window][:, None]     # (m, 1)

            # Rescale to local coordinate to reduce conditioning issues
            xi, xi_nodes = rescale_to_local(float(t), x_nodes, h=float(stride))

            val = neville_eval(xi, xi_nodes, y_nodes)  # returns shape (1,)
            f_recon[t] = val[0]

    # Diagnostics
    missing = ~is_sampled
    max_missing_error = np.max(np.abs(f_recon[missing] - f_full[missing]))
    max_sampled_error = np.max(np.abs(f_recon[is_sampled] - f_full[is_sampled]))
    print("max sampled error:", max_sampled_error)
    print("max missing error:", max_missing_error)

    # Local table around some point
    for t in range(12, 23):
        tag = "S" if is_sampled[t] else "M"
        err = abs(f_recon[t] - f_full[t])
        print(t, tag, f_full[t], f_recon[t], err)


if __name__ == "__main__":
    main()
