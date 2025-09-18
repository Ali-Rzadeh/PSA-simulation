import numpy as np

#from Isotherm import isotherm



def stream_composition_calculator_julia(time, state_vars, Params, ProductEnd="HPEnd"):
    """
    Python port of the Julia function.

    Parameters
    ----------
    time : (a,) array_like
        Time grid (1-D).
    state_vars : (a, n_state_vars) array_like
        State history; row i is the full state at time time[i].
    Params : sequence
    ProductEnd : {"HPEnd", "LPEnd"}
        Which end to evaluate.

    Returns
    -------
    n_tot : float
        Integrated total molar flow over 'time'.
    n_CO2 : float
        Integrated CO2 molar flow over 'time'.
    Temp : float
        Flow-weighted average temperature.
    """
   
    # --- Unpack Params

    N        = int(Params[0])
    L        = Params[17]
    R        = Params[8]
    T_0      = Params[4]
    P_0      = Params[16]
    MW_CO2   = Params[18]
    MW_N2    = Params[19]
    epsilon  = Params[5]
    mu       = Params[7]
    r_p      = Params[6]

    time = np.asarray(time)
    state_vars = np.asarray(state_vars)
    dz = L / N

    ndot_tot = np.empty_like(time, dtype=float)
    ndot_CO2 = np.empty_like(time, dtype=float)
    Temp_vec = np.empty_like(time, dtype=float)

    for i, tau in enumerate(time):
        x = state_vars[i, :]

        if ProductEnd == "HPEnd":

            P = x[0:2] * P_0
            y = x[N + 2]
            T = x[4*N + 8] * T_0
            P_node = P[0]  
        else:  # "LPEnd"
           
            P = x[N:N+2] * P_0
            y = x[2*N + 3]
            T = x[5*N + 9] * T_0
            P_node = P[1]  

        # Gas density and concentrations
        ro_g = (y * MW_CO2 + (1.0 - y) * MW_N2) * P_node / R / T
        C_tot = P_node / R / T
        C_CO2 = C_tot * y

        # Pressure gradient and velocity (Ergun-like)
        dPdz = 2.0 * (P[1] - P[0]) / dz
        viscous_term = 150.0 * mu * (1.0 - epsilon)**2 / 4.0 / (r_p**2) / (epsilon**3)
        kinetic_term = 1.75 * (1.0 - epsilon) / 2.0 / r_p / (epsilon**3) * ro_g

        if abs(kinetic_term) > 1e-10:
            v = -np.sign(dPdz) * (-viscous_term + np.sqrt(viscous_term**2 + 4.0 * kinetic_term * abs(dPdz))) / (2.0 * kinetic_term)
        else:
            v = 0.0

        ndot_tot[i] = abs(v * C_tot)
        ndot_CO2[i] = abs(v * C_CO2)
        Temp_vec[i] = T

    # Time integrals
    n_tot = np.trapezoid(ndot_tot, time)
    n_CO2 = np.trapezoid(ndot_CO2, time)


    energy_flux_tot = ndot_tot * Temp_vec
    energy_tot = np.trapezoid(energy_flux_tot, time)
    Temp = energy_tot / n_tot

    # Flow-weighted average temperature; fall back to simple mean if n_tot ~ 0
   # if n_tot > 0:
       # energy_flux_tot = ndot_tot * Temp_vec
      #  energy_tot = np.trapz(energy_flux_tot, time)
     #   Temp = energy_tot / n_tot
#    else:
  #      Temp = float(np.mean(Temp_vec))

    return n_tot, n_CO2, Temp




def stream_composition_calculator(time, state_vars, Params, ProductEnd):
    """
    Calculate the composition (total moles, CO2 moles, and temperature) at the ends of the PSA column.
    """
 
    # Unpack parameters from input
    L = Params[17]
    N = int(Params[0])
    P_0 = Params[16]
    T_0 = Params[4]
    MW_CO2 = Params[18]
    MW_N2 = Params[19]
    R = Params[8]
    mu = Params[7]
    epsilon = Params[5]
    r_p = Params[6]

    dz = L / N
   

    if ProductEnd.lower() == 'hpend':
        P = state_vars[:, 0:2] * P_0
        y = state_vars[:, N+2]  
        T = state_vars[:,4*N+8] * T_0  
        ro_g = (y * MW_CO2 + (1 - y) * MW_N2) * P[:, 0] / R / T
        C_tot = P[:, 0] / R / T
        C_CO2 = C_tot * y

    elif ProductEnd.lower() == 'lpend':
        P = state_vars[:, N:N+2] * P_0  
        y = state_vars[:, 2*N+3]        
        T = state_vars[:, 5*N+9] * T_0  
        ro_g = (y * MW_CO2 + (1 - y) * MW_N2) * P[:, 1] / R / T
        C_tot = P[:, 1] / R / T
        C_CO2 = C_tot * y

    else:
        raise ValueError('Please specify "HPEnd" or "LPEnd" for ProductEnd')

    # Pressure gradient [Pa/m]
    dPdz = 2 * (P[:, 1] - P[:, 0]) / dz
    #print(np.max(np.abs(P[:, 1] - P[:, 0])))  


    # Ergun equation (superficial velocity)
    viscous_term = 150 * mu * (1 - epsilon)**2 / 4 / r_p**2 / epsilon**3
    kinetic_term = (1.75 * (1 - epsilon) / 2 / r_p / epsilon**3) * ro_g
    v = -np.sign(dPdz) * (-viscous_term + (np.abs(viscous_term**2 + 4*kinetic_term*np.abs(dPdz))) **(0.5))/ 2/ kinetic_term


    # Molar fluxes [mol/m^2/s]
    ndot_tot = np.abs(v * C_tot)
    ndot_CO2 = np.abs(v * C_CO2)

    # Integrate over time
    n_tot = np.trapezoid(ndot_tot, time)
    n_CO2 = np.trapezoid(ndot_CO2, time)

    # Average temperature [K] (mole-weighted)
    energy_flux_tot = ndot_tot * T
    energy_tot = np.trapezoid(energy_flux_tot, time)
    Temp = energy_tot / n_tot 

    return n_tot, n_CO2, Temp



def velocity_correction(x, n_hr, Params, CorrectionEnd='HPEnd'):
    
     # Call the input function to set up parameters
    # Unpack parameters from input
    L = Params[17]
    N = int(Params[0])
    P_0 = Params[16]
    T_0 = Params[4]
    MW_CO2 = Params[18]
    MW_N2 = Params[19]
    R = Params[8]
    mu = Params[7]
    epsilon = Params[5]
    r_p = Params[6]
    """
    Correct the velocity at a given end of the column by adjusting pressure (Ergun-based).
    
    Parameters
    ----------
    x : ndarray
        State variable matrix (each row is a time step/point).
    n_hr : float or array-like
        Specified molar velocity for correction.
    CorrectionEnd : str
        'HPEnd' or 'LPEnd'.
    All other parameters: column/properties.
   
    """
    x_new = np.copy(x)
    N = int(N)
    dz = L / N

    if CorrectionEnd.lower() == 'hpend':
        T = x[:, 4*N+8] * T_0       
        y = x[:, N+2]               
        P = x[:, 1] * P_0           
    elif CorrectionEnd.lower() == 'lpend':
        T = x[:, 5*N+9] * T_0       
        y = x[:, 2*N+3]             
        P = x[:, N] * P_0           
    else:
        raise ValueError('CorrectionEnd must be "HPEnd" or "LPEnd"')

    MW = MW_N2 + (MW_CO2 - MW_N2) * y

    a_1 = 150 * mu * (1 - epsilon)**2 * dz / 2 / 4 / r_p**2 / epsilon**3 / R / T
    a_2_1 = 1.75 * (1 - epsilon) / 2 / r_p / epsilon**3 * dz / 2
    a_2 = a_2_1 / R / T * n_hr * MW

    a_a = a_1 + a_2
    b_b = P / T / R
    c_c = -n_hr

    vh = (-b_b + np.sqrt(b_b**2 - 4 * a_a * c_c)) / 2 / a_a

    a_p = a_1 * T * R
    b_p = a_2_1 * MW / R / T

    if CorrectionEnd.lower() == 'hpend':
        x_new[:, 0] = ((a_p * vh + P) / (1 - b_p * vh**2)) / P_0
    elif CorrectionEnd.lower() == 'lpend':
        x_new[:, N+1] = ((a_p * vh + P) / (1 - b_p * vh**2)) / P_0

    return x_new



def velocity_cleanup(x, Params):

     # Call the input function to set up parameters
    # Unpack parameters from input
    L = Params[17]
    R= Params[8]
    v_0 = Params[9]
    N = int(Params[0])
    P_0 = Params[16]
    T_0 = Params[4]
    MW_CO2 = Params[18]
    MW_N2 = Params[19]
   
    mu = Params[7]
    epsilon = Params[5]
    r_p = Params[6]

   



    """
    Adjust the entrance pressure using Ergun equation velocity balance.

    Parameters
    ----------
    x : ndarray
        State variable matrix (each row is a state at a time step).
    mu, epsilon, r_p, v_0, L, P_0, R, T_0, MW_N2, MW_CO2, N : float
        Process/column parameters.
   
       
    """

    x_new = np.copy(x)
    numb1 = 150 * mu * (1 - epsilon) ** 2 / 4 / r_p ** 2 / epsilon ** 2
    ro_gent = x[:, 1] * P_0 / R / T_0     
    numb2_ent = ro_gent * (MW_N2 + (MW_CO2 - MW_N2) * x[:, N+2]) * (1.75 * (1 - epsilon) / 2 / r_p / epsilon)
    

    # Update entrance pressure (x_new[:,0])
    x_new[:, 0] = (numb1 * v_0 + numb2_ent * v_0 ** 2) * L / P_0 / 2 / N + x[:, 1]

    return x_new




def process_evaluation(a, b, c, d, e, t1, t2, t3, t4, t5, Params):
    """
    Calculate the purity, recovery, and mass balance for the PSA cycle heavy product.

    Args:
        args: Should be (a, b, c, d, e, t1, t2, t3, t4, t5)
            a-e: State variable matrices (one per step)
            t1-t5: Time arrays (one per step)
        beta: Heavy reflux split fraction


    """
    #Params = np.asarray(Params)
    # Split args into state variables (step) and time vectors (tau)
    #nsteps = len(args) // 2

    v_0 = Params[9]
    L = Params[17]
    step = [a, b, c, d, e]
    tau  = [t1* L / v_0, t2* L / v_0, t3* L / v_0, t4* L / v_0, t5* L / v_0]
    beta = Params[30]

    # Extract n_tot, n_CO2, Temp for each needed stream 
    _, n_CO2_CoCPres_HPEnd, _     = stream_composition_calculator(tau[0], step[0], Params, 'HPEnd')
    _, n_CO2_ads_HPEnd, _         = stream_composition_calculator(tau[1], step[1], Params,'HPEnd')
    _, n_CO2_ads_LPEnd, _         = stream_composition_calculator(tau[1], step[1],Params, 'LPEnd')
    _, n_CO2_HR_LPEnd, _          = stream_composition_calculator(tau[2], step[2], Params, 'LPEnd')
    _, n_CO2_HR_HPEnd, _          = stream_composition_calculator(tau[2], step[2], Params, 'HPEnd')
    n_tot_CnCDepres_HPEnd, n_CO2_CnCDepres_HPEnd, _ = stream_composition_calculator(tau[3], step[3], Params, 'HPEnd')
    _, n_CO2_LR_LPEnd, _          = stream_composition_calculator(tau[4], step[4], Params, 'LPEnd')
    n_tot_LR_HPEnd, n_CO2_LR_HPEnd, _ = stream_composition_calculator(tau[4], step[4], Params, 'HPEnd')

    # Calculate purity, recovery, and mass balance 
    purity = (n_CO2_CnCDepres_HPEnd + (1 - beta) * n_CO2_LR_HPEnd) / (n_tot_CnCDepres_HPEnd + (1 - beta) * n_tot_LR_HPEnd)
    recovery = (n_CO2_CnCDepres_HPEnd + (1 - beta) * n_CO2_LR_HPEnd) / (n_CO2_CoCPres_HPEnd + n_CO2_ads_HPEnd)
    mass_balance = (n_CO2_CnCDepres_HPEnd + n_CO2_ads_LPEnd + n_CO2_HR_LPEnd + n_CO2_LR_HPEnd) / (n_CO2_CoCPres_HPEnd + n_CO2_ads_HPEnd + n_CO2_HR_HPEnd + n_CO2_LR_LPEnd)
    return purity, recovery, mass_balance



def CompressionEnergy(time, state_vars, r_in, Params, Patm = 1e5):
    """
    Calculate the compression energy for a given stream.

    Parameters
    ----------
    time : (a,) array_like
        Time grid (1-D).
    x : (a, n_state_vars) array_like
        State history; row i is the full state at time time[i].
    P : (a, 2) array_like
        Pressure at both ends of the column over time.

    Returns
    -------
    W_comp : float
        Compression work in Joules.
    """
   
    # Unpack parameters from input
    L = Params[17]
    N = int(Params[0])
    P_0 = Params[16]
    T_0 = Params[4]
    MW_CO2 = Params[18]
    MW_N2 = Params[19]
    R = Params[8]
    mu = Params[7]
    epsilon = Params[5]
    r_p = Params[6]
   
    dz = L / N

    adiabatic_index = 1.4
    compressor_efficiency = 0.72

    P = state_vars[:, 0:2] * P_0
    y = state_vars[:, N+2]  
    T = state_vars[:,4*N+8] * T_0   

    ro_g = (y * MW_CO2 + (1.0 - y) * MW_N2) * P[:,0] / R / T
    dPdz = 2.0 * (P[:, 1] - P[:, 0]) / dz

    viscous_term = 150.0 * mu * (1.0 - epsilon)**2 / 4.0 / (r_p**2) / (epsilon**3)
    kinetic_term = (1.75 * (1-epsilon) / 2.0 / r_p / (epsilon**3)) * ro_g

    #v = -np.sign(dPdz) * (-viscous_term +( np.abs(viscous_term**2 + 4.0 * kinetic_term * np.abs(dPdz))))**0.5 / (2.0 * kinetic_term)
    v = -np.sign(dPdz) * (-viscous_term + np.sqrt(np.abs(viscous_term**2 + 4.0*kinetic_term*np.abs(dPdz)))) / (2.0*kinetic_term)

    rati_term = ((P[:,0] / Patm)**((adiabatic_index - 1.0) / adiabatic_index) - 1.0)
    rati_term = np.maximum(rati_term, 0)  # Ensure non-negative

    integral_term = np.abs(v*P[:,0]* rati_term)

    energy = np.trapz(integral_term, time) * ((adiabatic_index) / (adiabatic_index - 1.0) )/ compressor_efficiency*np.pi*r_in **2


    return energy/3.6e6



def VacuumEnergy(time, state_vars, r_in,  ProductEnd, Params, Patm=1e5): 
    """
    Vacuum energy required over a step (kWh).

    Parameters
    ----------
    time : (nt,) array
        Dimensional time vector [s].
    state_vars : (nt, nstate) array
        Dimensionless state history for this step; each row is the full state at time[i].
    Patm : float
        Atmospheric pressure [Pa]; vacuum energy is only counted when P_out < Patm.
    ProductEnd : {"HPEnd","LPEnd"}, optional
        Which end to evaluate (heavy-product end or light-product end).
    Params : sequence, required
       
    r_in : float, required
        Column inner radius [m].

    Returns
    -------
    energy_kwh : float
        Total vacuum energy over the step [kWh].
    """
    if Params is None or r_in is None:
        raise ValueError("Params and r_in must be provided.")

    # --- Unpack parameters (match your other Python ports) ---
    N        = int(Params[0])
    T0       = Params[4]
    epsilon  = Params[5]
    r_p      = Params[6]
    mu       = Params[7]
    R        = Params[8]
    P0       = Params[16]
    L        = Params[17]
    MW_CO2   = Params[18]
    MW_N2    = Params[19]

    # Column differential length
    dz = L / N

    # Vacuum/polytropic parameters
    gamma = 1.4               # adiabatic index
    eta_vac = 0.72            # vacuum pump efficiency

    sv = np.asarray(state_vars)
    t  = np.asarray(time)
    ProductEnd ="hpend"

    # --- Pull end states (dimensioned) ---
    if ProductEnd.lower() == "hpend":
        # Left (heavy) end uses cells 1 and 2 for gradient
        P = sv[:, 0:2] * P0
        y = sv[:, N+2]
        T = sv[:, 4*N + 8] * T0
        P_out = P[:, 0]
    elif ProductEnd.lower() == "lpend":
        # Right (light) end uses cells N+1 and N+2 for gradient
        P = sv[:, N:N+2] * P0
        y = sv[:, 2*N + 3]
        T = sv[:, 5*N + 9] * T0
        P_out = P[:, 1]
    else:
        raise ValueError('ProductEnd must be "HPEnd" or "LPEnd".')

    # Gas density [kg/m^3] at outlet node
    MW_mix = y * MW_CO2 + (1.0 - y) * MW_N2
    rho_g = MW_mix * P_out / (R * T)

    # Pressure gradient at the end faces [Pa/m]
    dPdz = 2.0 * (P[:, 1] - P[:, 0]) / dz

    # Ergun superficial velocity [m/s]
    viscous_term = 150.0 * mu * (1.0 - epsilon)**2 / (4.0 * r_p**2 * epsilon**3)
    kinetic_term = (1.75 * (1.0 - epsilon) / (2.0 * r_p * epsilon**3)) * rho_g

    # Solve quadratic for |v| and apply sign(dPdz)
    # v = -sign(dPdz) * ( -a + sqrt(a^2 + 4 b |dPdz| ) ) / (2 b)
    a = viscous_term
    b = kinetic_term
    abs_dPdz = np.abs(dPdz)

    # Avoid division by ~0 when b ≈ 0
    v = np.zeros_like(P_out)
    mask = b > 1e-14
    v[mask] = -np.sign(dPdz[mask]) * (
        -a + np.sqrt(a*a + 4.0 * b[mask] * abs_dPdz[mask])
    ) / (2.0 * b[mask])

    # Compression ratio term (vacuum work factor); zero when P_out >= Patm
    expo = (gamma - 1.0) / gamma
    ratio_term = (Patm / P_out)**expo - 1.0
    ratio_term = np.maximum(ratio_term, 0.0)

    # Power density term (per cross-sectional area) to integrate over time
    integrand = np.abs(v * P_out * ratio_term)

    # Integrate and multiply by geometric/thermo factors
    # Energy [J] = ∫ (A * integrand * gamma/(gamma-1) / eta_vac) dt
    area = np.pi * r_in**2
    energy_J = np.trapz(integrand, t) * (gamma / (gamma - 1.0)) / eta_vac * area

    # Convert J -> kWh
    energy_kwh = energy_J / 3.6e6 
    return float(energy_kwh)

