import numpy as np
def build_time_grids(num_frames: int, stride: int):
    #we return t_full, t_sampled, t_recon in this order

    if num_frames <=0:
        raise ValueError("num_frames must be > 0")
    if stride <=0:
        raise ValueError("Stride must be > 0")
    
    t_full = np.arange(0,num_frames,1)
    t_recon = t_full.copy()
    

    t_sampled = np.arange(0,num_frames,stride)
    
    return t_full, t_sampled, t_recon