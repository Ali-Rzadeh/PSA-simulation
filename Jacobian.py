import numpy as np
from scipy.sparse import spdiags, bmat, csr_matrix

def jac_adsorption(N):
    """Constructs the sparsity pattern for the adsorption step Jacobian."""
    n = N + 2

    # Band matrix for advection terms
    B4 = np.ones((n, 4))
    A4 = spdiags(B4.T, [-2, -1, 0, 1], n, n).toarray()

    # Diagonal for adsorption term (except inlet/outlet)
    A1 = np.eye(n)
    A1[0, 0] = 0
    A1[-1, -1] = 0

    # Zero block
    A0 = np.zeros((n, n))

    # Block Jacobian assembly
    J_blocks = [
        [A4, A4, A1, A1, A4],
        [A4, A4, A1, A1, A4],
        [A1, A1, A1, A0, A1],
        [A1, A1, A0, A1, A1],
        [A4, A1, A1, A1, A4],
    ]
    J_ads = np.block(J_blocks)

    # Apply boundary conditions
    J_ads[0, :] = 0
    J_ads[:, 0] = 0

    J_ads[N+1, :] = 0
    J_ads[:, N+1] = 0

    J_ads[N+2, :] = 0
    J_ads[:, N+2] = 0

    J_ads[2*N+3, :] = J_ads[2*N+2, :]
    J_ads[:, 2*N+3] = 0

    J_ads[4*N+8, :] = 0
    J_ads[:, 4*N+8] = 0

    J_ads[5*N+9, :] = J_ads[5*N+8, :]
    J_ads[:, 5*N+9] = 0

    return csr_matrix(J_ads)

def jac_pressurization(N):
    """Jacobian sparsity pattern for the pressurization step ."""
    n = N + 2  # number of control volumes + 2 ghost cells

    # Four-band advection term (±2, ±1, 0)
    B4 = np.ones((n, 4))
    #A4 = spdiags(B4.T, [-2, -1, 0, 1], n, n).toarray()
    # replace A4 construction
    A4 = spdiags(
        [np.ones(n), np.ones(n), np.ones(n), np.ones(n), np.ones(n)],
        [-2, -1, 0, 1, 2],  # add +2
        n, n
    ).toarray()  # dense is fine if you’re ignoring efficiency


    # One-band diagonal term for LDF or source/sink term
    A1 = np.eye(n)
    A1[0, 0] = 0
    A1[-1, -1] = 0

    A0 = np.zeros((n, n))  # placeholder for zero blocks
    #ArithmeticError = np.zeros ((n,n))
    #jac_adsorption = J_pres[0,:]

    # Block assembly: 5 x 5 blocks of n x n
    J_blocks = [
        [A4, A4, A1, A1, A4],
        [A4, A4, A1, A1, A4],
        [A1, A1, A1, A0, A1],
        [A1, A1, A0, A1, A1],
        [A4, A1, A1, A1, A4],
    ]
    J_pres = np.block(J_blocks)

    # Apply boundary conditions 
    J_pres[0, :] = 0
    J_pres[0, 0] = 1                      # fixed pressure at inlet

    J_pres[N+1, :] = J_pres[N, :]      # pressure outlet = upstream node
    J_pres[:, N+1] = 0

    J_pres[N+2, :] = 0                     # y_inlet row
    J_pres[:, N+2] = 0

    J_pres[2*N+3, :] = J_pres[2*N+2, :]  # y_outlet = previous node
    J_pres[:, 2*N+3] = 0

    J_pres[4*N+8, :] = 0                 # T_inlet
    J_pres[:, 4*N+8] = 0

    J_pres[5*N+9, :] = J_pres[5*N+8, :]  # T_outlet = upstream
    J_pres[:, 5*N+9] = 0
# after your row edits, ensure these columns have at least the diagonal True
    for c in (n-1,      # P outlet ghost
            n,        # y inlet
            2*n - 1,  # y outlet ghost
            4*n,      # T inlet
            5*n - 1   # T outlet ghost
            ):
        J_pres[c, c] = 1
    # keep your row operations exactly as in MATLAB

    return csr_matrix(J_pres)



def jac_cnc_depressurization(N):
    """Jacobian sparsity pattern for counter-current depressurization step."""
    n = N + 2

    # Reversed advection terms: bands at 0, +1, +2 (and -1 for smoothing)
    B4 = np.ones((n, 4))
    A4 = spdiags(B4.T, [-1, 0, 1, 2], n, n).toarray()

    # Smooth inlet/outlet by copying neighbor rows
    A4[0, :] = A4[1, :]
    A4[-1, :] = A4[-2, :]

    # Adsorption terms: diagonal with special inlet/outlet coupling
    A1 = np.eye(n)
    A1[0, 0] = 0
    A1[-1, -1] = 0
    A1[0, 1] = 1
    A1[-1, -2] = 1

    A0 = np.zeros((n, n))

    # Build 5x5 block matrix
    J_blocks = [
        [A4, A4, A1, A1, A4],
        [A4, A4, A1, A1, A4],
        [A1, A1, A1, A0, A1],
        [A1, A1, A0, A1, A1],
        [A4, A1, A1, A1, A4],
    ]
    J_cnc_dep = np.block(J_blocks)

    # Apply boundary modifications 
    J_cnc_dep[0, :] = 0
    J_cnc_dep[0, 0] = 1  # pressure inlet fixed

    J_cnc_dep[N+1, :] = J_cnc_dep[N, :]         # pressure outlet = upstream
    J_cnc_dep[N+2, :] = J_cnc_dep[N+3, :]           # y_inlet = next cell
    J_cnc_dep[2*N+3, :] = J_cnc_dep[2*N+2, :]     # y_outlet = upstream

    # Molar loading rows zeroed
    J_cnc_dep[2*N+4, :] = 0
    J_cnc_dep[3*N+5:3*N+7, :] = 0
    J_cnc_dep[4*N+7, :] = 0

    # Temperature inlet = next cell
    J_cnc_dep[4*N+8, :] = J_cnc_dep[4*N+9, :]

    # Temperature outlet = upstream cell
    J_cnc_dep[5*N+9, :] = J_cnc_dep[5*N+8, :]

    return csr_matrix(J_cnc_dep)



def jac_light_reflux(N):
    """Jacobian sparsity pattern for the light reflux step."""
    n = N + 2

    # Advection direction: downstream → upstream (right to left)
    B4 = np.ones((n, 4))
    A4 = spdiags(B4.T, [-1, 0, 1, 2], n, n).toarray()
    A4[0, :] = A4[1, :]
    A4[-1, :] = A4[-2, :]

    # Adsorption/Desorption pattern with extra coupling at inlet/outlet
    A1 = np.eye(n)
    A1[0, 0] = 0
    A1[-1, -1] = 0
    A1[0, 1] = 1
    A1[-1, -2] = 1

    A0 = np.zeros((n, n))

    # Assemble 5x5 block matrix
    J_blocks = [
        [A4, A1, A1, A1, A4],
        [A4, A4, A1, A1, A4],
        [A1, A1, A1, A0, A1],
        [A1, A1, A0, A1, A1],
        [A4, A1, A1, A1, A4],
    ]
    J_lr = np.block(J_blocks)

    # Boundary condition mappings
    J_lr[0, :] = 0
    J_lr[0, 0] = 1                         # pressure inlet fixed

    J_lr[N+1, :] = J_lr[N, :]              # pressure outlet
    J_lr[N+2, :] = J_lr[N+3, :]                # mole fraction inlet
    J_lr[2*N+3, :] = J_lr[2*N+2, :]          # mole fraction outlet

    J_lr[2*N+4, :] = 0                    # molar loading
    J_lr[3*N+5:3*N+7, :] = 0
    J_lr[4*N+7, :] = 0

    J_lr[4*N+8, :] = J_lr[4*N+9, :]          # temperature inlet
    J_lr[5*N+9, :] = J_lr[5*N+8, :]          # temperature outlet

    return csr_matrix(J_lr)
