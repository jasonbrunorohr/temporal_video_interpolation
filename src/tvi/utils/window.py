
import numpy as np
def centered_window_indices(k_total: int, k_center: int, m: int) -> np.ndarray:
    """
    Return m indices in [0, k_total-1] that form a centered window around k_center.
    If near boundaries, clamp/shift so you still return exactly m indices.
    """
    if m<=0 or m> k_total:
        raise ValueError("change window size or k_total")
    
    k_center = min(max(k_center,0), k_total-1)
    k_start = k_center - m//2
    k_end = k_start + m -1
    if k_start < 0:
        k_start = 0
        k_end = m-1
    if k_end > k_total-1:
        k_end = k_total-1
        k_start = k_end -(m-1)
    indices = np.arange(k_start,k_end+1,1)
    return indices 
