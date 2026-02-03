import numpy as np

from tvi.utils.window import centered_window_indices


def chebyshev_points(m: int) -> np.ndarray:
    """
    Chebyshev points of the first kind (roots) mapped to [-1, 1]:
        c_k = cos((2k+1)π/(2m)), k=0..m-1
    """
    if m <= 0:
        raise ValueError("m must be positive")
    k = np.arange(m, dtype=np.float64)
    return np.cos((2.0 * k + 1.0) * np.pi / (2.0 * m))


def select_chebyshev_indices(k_total: int, k_center: int, m: int, M_avail: int) -> np.ndarray:
    """
    Choose m indices from a candidate window of size M_avail around k_center,
    approximately distributed like Chebyshev nodes (clustered near ends).

    Returns:
        idx_sel: shape (m,), strictly increasing integer indices in [0, k_total-1]
    """
    if k_total <= 0:
        raise ValueError("k_total must be positive")
    if m <= 0 or m > k_total:
        raise ValueError("m must satisfy 1 <= m <= k_total")
    if M_avail <= 0 or M_avail > k_total:
        raise ValueError("M_avail must satisfy 1 <= M_avail <= k_total")
    if m > M_avail:
        raise ValueError("need M_avail >= m")

    # Candidate indices (monotone, clamped to boundaries)
    cand = centered_window_indices(k_total=k_total, k_center=k_center, m=M_avail)  # (M_avail,)

    # Candidate positions mapped to [-1, 1] by rank
    if M_avail == 1:
        return cand.copy()  # only possible selection

    r = np.arange(M_avail, dtype=np.float64)
    s = -1.0 + 2.0 * r / (M_avail - 1.0)  # (M_avail,)

    # Chebyshev targets in [-1, 1]
    c = chebyshev_points(m)  # (m,)

    # Greedy nearest matching with uniqueness
    used = np.zeros(M_avail, dtype=bool)
    chosen_ranks = []

    for ck in c:
        # distances to each candidate position
        d = np.abs(s - ck)

        # pick nearest unused candidate
        order = np.argsort(d)
        pick = None
        for rr in order:
            if not used[rr]:
                pick = int(rr)
                break
        if pick is None:
            # fallback: should not happen if M_avail >= m
            break

        used[pick] = True
        chosen_ranks.append(pick)

    chosen_ranks = np.array(chosen_ranks, dtype=int)
    idx_sel = cand[chosen_ranks]
    idx_sel = np.unique(idx_sel)  # safety; should already be unique

    if idx_sel.size != m:
        # In rare boundary/clamping cases, duplicates could arise; repair by filling with unused ranks
        remaining = [i for i in range(M_avail) if not used[i]]
        # Add until we have m
        for rr in remaining:
            idx_sel = np.unique(np.append(idx_sel, cand[rr]))
            if idx_sel.size == m:
                break

    if idx_sel.size != m:
        raise RuntimeError("failed to select m unique indices; increase M_avail or check windowing")

    return np.sort(idx_sel)
