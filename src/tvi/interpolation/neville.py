import numpy as np

def neville_eval(x: float, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    """
    Neville (Aitken–Neville) evaluation of the interpolating polynomial at scalar x.

    Parameters
    ----------
    x : float
        Evaluation point.
    x_nodes : np.ndarray, shape (m,)
        Distinct nodes.
    y_nodes : np.ndarray, shape (m, P)
        Values at nodes (vector-valued allowed).

    Returns
    -------
    np.ndarray, shape (P,)
        Interpolated value at x.
    """
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    y_nodes = np.asarray(y_nodes, dtype=np.float64)

    if x_nodes.ndim != 1:
        raise ValueError("x_nodes must be 1D")
    if y_nodes.ndim != 2:
        raise ValueError("y_nodes must be 2D (m, P)")
    m = x_nodes.size
    if y_nodes.shape[0] != m:
        raise ValueError("y_nodes.shape[0] must equal len(x_nodes)")
    if m == 0:
        raise ValueError("need at least one node")

    # Exact-node shortcut (avoid division by zero)
    hits = np.where(x_nodes == x)[0]
    if hits.size > 0:
        return y_nodes[int(hits[0])]

    # Q will store the Neville table, but we only keep current column
    Q = y_nodes.copy()  # shape (m, P)

    # Build triangular scheme
    for j in range(1, m):
        for i in range(0, m - j):
            xi = x_nodes[i]
            xj = x_nodes[i + j]
            denom = (xj - xi)
            if denom == 0.0:
                raise ValueError("duplicate nodes detected in x_nodes")
            Q[i] = ((x - xi) * Q[i + 1] - (x - xj) * Q[i]) / denom

    return Q[0]
