import numpy as np
from DDM.utls.utl_ks import utl_ks

def utl_fsw(t,w,prec):
    K = utl_ks(t,w,prec)
    f = np.zeros_like(t)    # Initialize density array with the same shape as `t`
    if np.all(K > 0) and np.all(np.isfinite(K)):
        for k in range(K,0, -1):
            term1 = (w + 2 * k) * np.exp(-((w + 2 * k)**2) / (2 * t))
            term2 = (w - 2 * k) * np.exp(-((w - 2 * k)**2) / (2 * t))
            f += np.array(term1) + np.array(term2)
        
        # Add final component to `f`
        f = (1 / np.sqrt(2 * np.pi * t**3)) * (f + w * np.exp(-(w**2) / (2 * t)))
    
    return f