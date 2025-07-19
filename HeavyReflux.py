import numpy as np
from WENO import weno
from Isotherm import isotherm

print('Running HeavyReflux function...')    

def func_heavy_reflux(t, state_vars, params, isotherm_params):
    N        = int(params[0])
    deltaU_1 = params[1]
    deltaU_2 = params[2]
    ro_s     = params[3]
    T_0      = params[4]
    epsilon  = params[5]
    r_p      = params[6]
    mu       = params[7]
    R        = params[8]
    v_0      = params[9]
    q_s0     = params[10]
    C_pg     = params[11]
    C_pa     = params[12]
    C_ps     = params[13]
    D_m      = params[14]
    K_z      = params[15]
    P_0      = params[16]
    L        = params[17]
    MW_CO2   = params[18]
    MW_N2    = params[19]
    k_1_LDF  = params[20]
    k_2_LDF  = params[21]
    P_inlet  = params[25]
    y_HR     = params[32]
    T_HR     = params[33]
    ndot_HR  = params[34]

    # Initialize state variables
    P  = state_vars[0:N+2]
    y  = np.maximum(state_vars[N+2:2*N+4], 0)
    x1 = np.maximum(state_vars[2*N+4:3*N+6], 0)
    x2 = state_vars[3*N+6:4*N+8]
    T  = state_vars[4*N+8:5*N+10]

    derivatives = np.zeros(5*N+10)
    dPdt  = np.zeros(N+2)
    dPdt1 = np.zeros(N+2)
    dPdt2 = np.zeros(N+2)
    dPdt3 = np.zeros(N+2)
    dydt  = np.zeros(N+2)
    dydt1 = np.zeros(N+2)
    dydt2 = np.zeros(N+2)
    dydt3 = np.zeros(N+2)
    dx1dt = np.zeros(N+2)
    dx2dt = np.zeros(N+2)
    dTdt  = np.zeros(N+2)
    dTdt1 = np.zeros(N+2)
    dTdt2 = np.zeros(N+2)
    dTdt3 = np.zeros(N+2)
    dpdz   = np.zeros(N+2)
    dpdzh  = np.zeros(N+1)
    dydz   = np.zeros(N+2)
    d2ydz2 = np.zeros(N+2)
    dTdz   = np.zeros(N+2)
    d2Tdz2 = np.zeros(N+2)

    dz = 1.0 / N
    D_l  = 0.7 * D_m + v_0 * r_p
    Pe   = v_0 * L / D_l
    phi  = R * T_0 * q_s0 * (1-epsilon) / epsilon / P_0
    ro_g = P * P_0 / R / T / T_0

    # Boundary conditions
    y[0] = y_HR
    T[0] = T_HR / T_0

    if params[-1] == 1:
        P[0] = P_inlet
    elif params[-1] == 0:
        vh = np.zeros(N+1)
        MW = MW_N2 + (MW_CO2 - MW_N2) * y[0]

        a_1 = 150*mu*(1-epsilon)**2*dz*L/2/4/r_p**2/epsilon**3/T[0]/T_0/R
        a_2_1 = 1.75*(1-epsilon)/2/r_p/epsilon**3*dz*L/2
        a_2 = a_2_1/R/T[0]/T_0*ndot_HR*MW
        a = a_1 + a_2
        b = P[1] / T[0] * P_0 / R / T_0
        c = -ndot_HR
        vh[0] = (-b + np.sqrt(b**2 - 4*a*c)) / 2/a/v_0
        a_p = a_1 * T[0] * T_0 * R
        b_p = a_2_1 * MW / R / T[0] / T_0
        P[0] = ((a_p * vh[0] * v_0 + P[1]*P_0) / (1 - b_p * vh[0]**2 * v_0**2)) / P_0
    else:
        raise ValueError('Specify whether inlet velocity or pressure is constant for the feed step')

    y[-1] = y[-2]
    T[-1] = T[-2]
    if P[-2] >= 1:
        P[-1] = 1
    else:
        P[-1] = P[-2]

    # Spatial derivatives via WENO 
    Ph = weno(P, 'upwind')
    dpdz[1:-1] = (Ph[1:] - Ph[:-1]) / dz
    dpdzh[1:N] = (P[2:N+1] - P[1:N]) / dz
    dpdzh[0] = 2 * (P[1] - P[0]) / dz
    dpdzh[N] = 2 * (P[N+1] - P[N]) / dz

    yh = weno(y, 'upwind')
    dydz[1:N+1] = (yh[1:N+1] - yh[0:N]) / dz

    Th = weno(T, 'upwind')
    dTdz[1:N+1] = (Th[1:N+1] - Th[0:N]) / dz

    # Second derivatives (central differences)
    d2ydz2[2:N] = (y[3:N+1] + y[1:N-1] - 2*y[2:N]) / dz**2
    d2ydz2[1] = (y[2] - y[1]) / dz**2
    d2ydz2[N] = (y[N-1] - y[N]) / dz**2

    d2Tdz2[2:N] = (T[3:N+1] + T[1:N-1] - 2*T[2:N]) / dz**2
    d2Tdz2[1] = 4 * (Th[1] + T[0] - 2*T[1]) / dz**2
    d2Tdz2[N] = 4 * (Th[N-1] + T[N+1] - 2*T[N]) / dz**2

    # Velocity at faces
    ro_gh = (P_0 / R / T_0) * Ph / Th
    viscous_term = 150 * mu * (1-epsilon)**2 / 4 / r_p**2 / epsilon**2
    kinetic_term_h = (ro_gh * (MW_N2 + (MW_CO2 - MW_N2) * yh)) * (1.75 * (1-epsilon) / 2 / r_p / epsilon)
    vh = -np.sign(dpdzh) * (-viscous_term + (np.abs(viscous_term**2 + 4*kinetic_term_h*np.abs(dpdzh)*P_0/L)) **(0.5))/ 2/ kinetic_term_h/v_0
    # LDF Mass transfer and isotherm (assume Isotherm implemented)
    q = isotherm(y, P * P_0, T * T_0, isotherm_params)
    q_1 = q[:, 0] * ro_s
    q_2 = q[:, 1] * ro_s
    k_1 = k_1_LDF * L / v_0
    k_2 = k_2_LDF * L / v_0
    dx1dt[1:N+1] = k_1 * (q_1[1:N+1] / q_s0 - x1[1:N+1])
    dx2dt[1:N+1] = k_2 * (q_2[1:N+1] / q_s0 - x2[1:N+1])

    # Energy balance (column T)
    sink_term = (1-epsilon)*(ro_s*C_ps + q_s0*C_pa) + (epsilon * ro_g[1:N+1] * C_pg)
    transfer_term = K_z / v_0 / L
    dTdt1[1:N+1] = transfer_term * d2Tdz2[1:N+1] / sink_term

    PvT = Ph * vh / Th
    Pv = Ph * vh
    dTdt2[1:N+1] = -epsilon * C_pg * P_0 / R / T_0 * (
        (Pv[1:N+1] - Pv[0:N]) - T[1:N+1] * (PvT[1:N+1] - PvT[0:N])
    ) / dz / sink_term

    generation_term_1 = (1-epsilon) * q_s0 * (-(deltaU_1 - R * T[1:N+1] * T_0)) / T_0
    generation_term_2 = (1-epsilon) * q_s0 * (-(deltaU_2 - R * T[1:N+1] * T_0)) / T_0
    dTdt3[1:N+1] = (generation_term_1 * dx1dt[1:N+1] + generation_term_2 * dx2dt[1:N+1]) / sink_term
    dTdt[1:N+1] = dTdt1[1:N+1] + dTdt2[1:N+1] + dTdt3[1:N+1]

    # Total mass balance (pressure)
    dPdt1[1:N+1] = -T[1:N+1] * (PvT[1:N+1] - PvT[0:N]) / dz
    dPdt2[1:N+1] = -phi * T[1:N+1] * (dx1dt[1:N+1] + dx2dt[1:N+1])
    dPdt3[1:N+1] = P[1:N+1] * dTdt[1:N+1] / T[1:N+1]
    dPdt[1:N+1] = dPdt1[1:N+1] + dPdt2[1:N+1] + dPdt3[1:N+1]

    # Component mass balance (mole fraction)
    dydt1[1:N+1] = (1/Pe) * (d2ydz2[1:N+1] + (dydz[1:N+1] * dpdz[1:N+1] / P[1:N+1]) - (dydz[1:N+1] * dTdz[1:N+1] / T[1:N+1]))
    ypvt = yh * Ph * vh / Th
    dydt2[1:N+1] = -(T[1:N+1] / P[1:N+1]) * (
        (ypvt[1:N+1] - ypvt[0:N]) - y[1:N+1] * (PvT[1:N+1] - PvT[0:N])
    ) / dz
    dydt3[1:N+1] = (phi * T[1:N+1] / P[1:N+1]) * (
        (y[1:N+1] - 1) * dx1dt[1:N+1] + y[1:N+1] * dx2dt[1:N+1]
    )
    dydt[1:N+1] = dydt1[1:N+1] + dydt2[1:N+1] + dydt3[1:N+1]

    # Boundary derivatives
    dPdt[0] = 0
    dPdt[N+1] = 0
    dydt[0] = 0
    dydt[N+1] = dydt[N]
    dx1dt[0] = 0
    dx2dt[0] = 0
    dx1dt[-1] = 0
    dx2dt[-1] = 0
    dTdt[0] = 0
    dTdt[-1] = dTdt[-2]

    # Export to output
    derivatives[0:N+2] = dPdt
    derivatives[N+2:2*N+4] = dydt
    derivatives[2*N+4:3*N+6] = dx1dt
    derivatives[3*N+6:4*N+8] = dx2dt
    derivatives[4*N+8:5*N+10] = dTdt

    return derivatives
