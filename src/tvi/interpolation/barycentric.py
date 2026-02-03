import numpy as np
def barycentric_weights(x_nodes: np.ndarray) -> np.ndarray:
    
    nodes = x_nodes.copy().astype('float64')
    m = len(x_nodes)
    w = np.empty(m, dtype=np.float64)

    for k in range(0,m):
        diff = nodes[k] - nodes
        diff[k] = 1.0
        w[k] = np.prod(diff)

    return 1/w 



def barycentric_eval(x: float,
                     x_nodes: np.ndarray,
                     y_nodes: np.ndarray,
                     w: np.ndarray) -> np.ndarray:
    """
    x_nodes: shape (m,)
    y_nodes: shape (m, P)
    w:       shape (m,)
    returns: shape (P,)
    """
    shapeP = len(y_nodes[0])
    hits = np.where(x_nodes == x)[0]
    if hits.size > 0:
        j = int(hits[0])
        return y_nodes[j]

   

    numerator = np.zeros(shapeP)
    denominator = 0
    m = len(x_nodes)
    for j in range(m):
        numerator += w[j]*y_nodes[j]/(x-x_nodes[j])
        denominator += w[j]/(x-x_nodes[j])

    return numerator/denominator 
