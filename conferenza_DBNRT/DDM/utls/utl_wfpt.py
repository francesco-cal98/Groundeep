from DDM.utls.utl_fsw import utl_fsw
import numpy as np 

def utl_wfpt(t, v, a, w=0.5, prec=1e-4):
    """
    First passage time for Wiener diffusion model.
    
    Args:
    t (float or np.ndarray): Hitting time (response time in milliseconds).
    v (float): Drift rate.
    a (float): Threshold.
    w (float): Bias (default: 0.5).
    prec (float): Error threshold (default: 1e-4).
    
    Returns:
    np.ndarray: Probability density at the lower barrier.
    """
    # Calculate the main density `p`
    p = (1 / (a**2)) * np.exp(-v * a * w - (v**2) * t / 2)
    # Adjust `p` by calling `utl_fsw`
    p *= utl_fsw(t / (a**2), w, prec / p)
    
    return p