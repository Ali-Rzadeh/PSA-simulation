import numpy as np

def weno_python(flux_c, flow_dir):
    """
    Weighted Essentially Non-Oscillatory (WENO) wall reconstruction for PSA FVM.
    Parameters:
        flux_c : (N+2,) array -- variable at cell centers
        flow_dir : 'upwind' or 'downwind'
    Returns:
        flux_w : (N+1,) array -- variable at cell faces/walls
    """
    oo = 1e-10
    N = flux_c.shape[0] - 2
    flux_w = np.zeros(N+1)
    alpha0 = np.zeros_like(flux_c)
    alpha1 = np.zeros_like(flux_c)

    # Boundary assignment
    flux_w[0]   = flux_c[0]
    flux_w[-1]  = flux_c[-1]

    if flow_dir.lower() == 'upwind':
        # Compute smoothness indicators
        alpha0[1:N]   = (2/3) / ( (flux_c[2:N+1] - flux_c[1:N] + oo) ** 4 )
        alpha1[2:N]   = (1/3) / ( (flux_c[2:N]   - flux_c[1:N-1] + oo) ** 4 )
        alpha1[1]     = (1/3) / ( (2*(flux_c[1]  - flux_c[0]) + oo) ** 4 )

        # Interior faces 
        for j in range(2, N):
            denom = alpha0[j] + alpha1[j]
            f0 = 0.5 * (flux_c[j] + flux_c[j+1])
            f1 = 1.5 * flux_c[j] - 0.5 * flux_c[j-1]
            flux_w[j] = (alpha0[j]/denom) * f0 + (alpha1[j]/denom) * f1

        # Left face just inside boundary 
        denom = alpha0[1] + alpha1[1]
        f0 = 0.5 * (flux_c[1] + flux_c[2])
        f1 = 2 * flux_c[1] - flux_c[0]
        flux_w[1] = (alpha0[1]/denom) * f0 + (alpha1[1]/denom) * f1

    elif flow_dir.lower() == 'downwind':
        alpha0[1:N]   = (2/3) / ( (flux_c[1:N] - flux_c[2:N+1] + oo) ** 4 )
        alpha1[1:N-1] = (1/3) / ( (flux_c[2:N] - flux_c[3:N+1] + oo) ** 4 )
        alpha1[N-1]   = (1/3) / ( (2*(flux_c[N] - flux_c[N+1]) + oo) ** 4 )

        # Interior faces 
        for j in range(1, N-1):
            denom = alpha0[j] + alpha1[j]
            f0 = 0.5 * (flux_c[j] + flux_c[j+1])
            f1 = 1.5 * flux_c[j+1] - 0.5 * flux_c[j+2]
            flux_w[j] = (alpha0[j]/denom) * f0 + (alpha1[j]/denom) * f1

        # Right face just inside boundary 
        denom = alpha0[N-1] + alpha1[N-1]
        f0 = 0.5 * (flux_c[N-1] + flux_c[N])
        f1 = 2 * flux_c[N] - flux_c[N+1]
        flux_w[N-1] = (alpha0[N-1]/denom) * f0 + (alpha1[N-1]/denom) * f1

    else:
        raise ValueError("flow_dir must be 'upwind' or 'downwind'")

    return flux_w





def weno(flux_c, flow_dir):
    flow_dir = flow_dir.lower()  
    oo = np.array(1e-10, dtype=getattr(flux_c, "dtype", float))

  
    fc = flux_c
    if fc.ndim == 1:
        fc = fc[:, None]
    N = fc.shape[0] - 2
    m = fc.shape[1]

    fw = np.zeros((N+1, m), dtype=fc.dtype)
    a0 = np.zeros_like(fc)
    a1 = np.zeros_like(fc)

    # boundary walls
    fw[0,  :] = fc[0,  :]
    fw[-1, :] = fc[-1, :]

    if flow_dir == 'upwind':
        a0[1:N,   :] = (2/3) / ((fc[2:N+1, :] - fc[1:N,   :] + oo)**4)
        a1[2:N,   :] = (1/3) / ((fc[2:N,   :] - fc[1:N-1, :] + oo)**4)
        a1[1,     :] = (1/3) / ((2*(fc[1,  :] - fc[0,    :]) + oo)**4)

        j = np.arange(2, N)
        denom = a0[j, :] + a1[j, :]
        f0 = 0.5*(fc[j, :] + fc[j+1, :])
        f1 = 1.5*fc[j, :] - 0.5*fc[j-1, :]
        fw[j, :] = (a0[j, :]/denom)*f0 + (a1[j, :]/denom)*f1

        denom = a0[1, :] + a1[1, :]
        f0 = 0.5*(fc[1, :] + fc[2, :])
        f1 = 2*fc[1, :] - fc[0, :]
        fw[1, :] = (a0[1, :]/denom)*f0 + (a1[1, :]/denom)*f1

    elif flow_dir == 'downwind':
        a0[1:N,   :] = (2/3) / ((fc[1:N,   :] - fc[2:N+1, :] + oo)**4)
        a1[1:N-1, :] = (1/3) / ((fc[2:N,   :] - fc[3:N+1, :] + oo)**4)
        a1[N-1,   :] = (1/3) / ((2*(fc[N,  :] - fc[N+1,  :]) + oo)**4)

        j = np.arange(1, N-1)
        denom = a0[j, :] + a1[j, :]
        f0 = 0.5*(fc[j, :] + fc[j+1, :])
        f1 = 1.5*fc[j+1, :] - 0.5*fc[j+2, :]
        fw[j, :] = (a0[j, :]/denom)*f0 + (a1[j, :]/denom)*f1

        denom = a0[N-1, :] + a1[N-1, :]
        f0 = 0.5*(fc[N-1, :] + fc[N, :])
        f1 = 2*fc[N, :] - fc[N+1, :]
        fw[N-1, :] = (a0[N-1, :]/denom)*f0 + (a1[N-1, :]/denom)*f1
    else:
        raise ValueError("flow_dir must be 'upwind' or 'downwind'")

    return fw if flux_c.ndim == 2 else fw.ravel()
