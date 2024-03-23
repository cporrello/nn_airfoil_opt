import numpy as np


"""Generate profile for NACA airfoil.

Assumes that points x begin at the TE and end at the TE, counter-clockwise """
def symmetric_NACA_4digit(x: np.ndarray, t=12):
    
    # Equation of line
    y = 5 * (t/100.0)*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x** 3 - 0.1036*x**4)
    
    # We need to set y for x past the LE to negative (the LE is the smallest x values)
    if x.shape[0] > 1:
        y[x.argmin()+1:] *= -1
    
    return y