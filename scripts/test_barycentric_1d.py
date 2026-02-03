import numpy as np

from tvi.utils.time_grid import build_time_grids
from tvi.utils.window import centered_window_indices
from tvi.interpolation.barycentric import barycentric_weights, barycentric_eval



t_full, t_sampled, t_recon = build_time_grids(num_frames=100, stride=5)

f_full = np.sin(0.8*t_full) + 1 + (t_full >= 50).astype(float)


f_sampled = f_full[t_sampled]


f_recon = np.empty_like(f_full)
m  = 4
is_sampled = np.zeros_like(f_full, dtype=bool)
is_sampled[t_sampled] = True
for t in t_recon:
    

    if is_sampled[t]:
        f_recon[t] = f_full[t]
    else:
        k_center = int(np.argmin(np.abs((t_sampled-t))))
        assert abs(t_sampled[k_center] - t) == np.min(np.abs(t_sampled - t))
        

        window = centered_window_indices(len(t_sampled),k_center,m)
        x_nodes = t_sampled[window]
        y_nodes = f_sampled[window]
        w = barycentric_weights(x_nodes)
        val = barycentric_eval(float(t),x_nodes, y_nodes[:,None],w)
        f_recon[t] = val[0]
    if t in [1,2,3,4]:
        print("t", t, "k_center", k_center, "x_nodes", x_nodes)





