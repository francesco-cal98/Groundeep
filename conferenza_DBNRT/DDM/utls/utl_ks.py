import numpy as np

def utl_ks(t, w, prec):
    """
    Calculates the number of terms for the density, based on Gondan, Blurton, and Kesselmeier (2014).
    
    Parameters:
        t (float or np.ndarray): Hitting time (e.g., response time).
        w (float): Bias parameter.
        prec (float): Precision or error threshold.
        
    Returns:
        np.ndarray: Number of terms (K) for each element in t.
    """
    # Initial computation for K1
    K1 = (np.sqrt(2 * t) - w) / 2
    K2 = np.copy(K1)  # Start K2 as a copy of K1

    # Calculate u_eps based on min(log(2 * pi * t^2 * prec^2), -1)
    u_eps = np.minimum(-1, np.log(2 * np.pi * t**2 * prec**2))
    
    # Calculate argument for K2 conditionally
    arg = -t * (u_eps - np.sqrt(-2 * u_eps - 2))
    K2[arg > 0] = 0.5 * np.sqrt(arg[arg > 0]) - w / 2

    # Return the ceiling of the max between K1 and K2
    K = int(np.ceil(np.max([K1, K2])))
    return K