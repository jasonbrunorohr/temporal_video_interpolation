import numpy as np

def rescale_to_local(x: float, x_nodes: np.ndarray, h: float | None = None) -> tuple[float, np.ndarray]:
    """
    Map x and x_nodes to a local coordinate:
        xi      = (x - x_c) / h
        xi_nodes= (x_nodes - x_c) / h

    where x_c is chosen as the middle node x_nodes[m//2].
    h is chosen as spacing between consecutive nodes if not provided.

    Returns:
        xi (float), xi_nodes (np.ndarray shape (m,))
    """
    x_nodes = np.asarray(x_nodes, dtype=np.float64)

    if x_nodes.ndim != 1 or x_nodes.size < 2:
        raise ValueError("x_nodes must be 1D with length >= 2")

    # Center point: middle node (stable and deterministic)
    xc = float(x_nodes[x_nodes.size // 2])

    # Scale: default to local spacing
    if h is None:
        h = float(x_nodes[1] - x_nodes[0])
        if h == 0.0:
            raise ValueError("x_nodes spacing is zero; nodes are not distinct")

    xi = (float(x) - xc) / h
    xi_nodes = (x_nodes - xc) / h
    return xi, xi_nodes
