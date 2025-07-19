import numpy as np
from Input import input
#from Isotherm import isotherm

def stream_composition_calculator(time, state_vars, ProductEnd):
    """
    Calculate the composition (total moles, CO2 moles, and temperature) at the ends of the PSA column.
    """
    Params = input()  # Call the input function to set up parameters
    # Unpack parameters from input
    L = Params[17]
    N = int(Params[0])
    P_0 = Params[16]
    T_0 = Params[4]
    MW_CO2 = Params[18]
    MW_N2 = Params[22]
    R = Params[19]
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
    print(np.max(np.abs(P[:, 1] - P[:, 0])))  


    # Ergun equation (superficial velocity)
    viscous_term = 150 * mu * (1 - epsilon)**2 / 4 / r_p**2 / epsilon**3
    kinetic_term = (1.75 * (1 - epsilon) / 2 / r_p / epsilon**3) * ro_g
    v = -np.sign(dPdz) * (-viscous_term + (np.abs(viscous_term**2 + 4*kinetic_term*np.abs(dPdz)*P_0/L)) **(0.5))/ 2/ kinetic_term

    # Molar fluxes [mol/m^2/s]
    ndot_tot = np.abs(v * C_tot)
    ndot_CO2 = np.abs(v * C_CO2)

    # Integrate over time
    n_tot = np.trapezoid(time, ndot_tot)
    n_CO2 = np.trapezoid(ndot_CO2, time)

    # Average temperature [K] (mole-weighted)
    energy_flux_tot = ndot_tot * T
    energy_tot = np.trapezoid(energy_flux_tot, time)
    Temp = energy_tot / n_tot 

    return n_tot, n_CO2, Temp



def velocity_correction(x, n_hr, CorrectionEnd='HPEnd'):
    
    Params = input()  # Call the input function to set up parameters
    # Unpack parameters from input
    L = Params[17]
    N = Params[0]
    P_0 = Params[16]
    T_0 = Params[4]
    MW_CO2 = Params[18]
    MW_N2 = Params[22]
    R = Params[19]
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



def velocity_cleanup(x):

    Params = input()  # Call the input function to set up parameters
    # Unpack parameters from input
    L = Params[17]
    R= Params[8]
    v_0 = Params[9]
    N = Params[0]
    P_0 = Params[16]
    T_0 = Params[4]
    MW_CO2 = Params[18]
    MW_N2 = Params[22]
    R = Params[19]
    mu = Params[7]
    epsilon = Params[5]
    r_p = Params[6]

    N = int(N)



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




def process_evaluation(*args, beta=0.0):
    """
    Calculate the purity, recovery, and mass balance for the PSA cycle heavy product.

    Args:
        args: Should be (a, b, c, d, e, t1, t2, t3, t4, t5)
            a-e: State variable matrices (one per step)
            t1-t5: Time arrays (one per step)
        beta: Heavy reflux split fraction


    """
    # Split args into state variables (step) and time vectors (tau)
    nsteps = len(args) // 2
    step = args[:nsteps]
    tau  = args[nsteps:]

    # Extract n_tot, n_CO2, Temp for each needed stream 
    _, n_CO2_CoCPres_HPEnd, _     = stream_composition_calculator(tau[0], step[0], 'HPEnd')
    _, n_CO2_ads_HPEnd, _         = stream_composition_calculator(tau[1], step[1], 'HPEnd')
    _, n_CO2_ads_LPEnd, _         = stream_composition_calculator(tau[1], step[1], 'LPEnd')
    _, n_CO2_HR_LPEnd, _          = stream_composition_calculator(tau[2], step[2], 'LPEnd')
    _, n_CO2_HR_HPEnd, _          = stream_composition_calculator(tau[2], step[2], 'HPEnd')
    n_tot_CnCDepres_HPEnd, n_CO2_CnCDepres_HPEnd, _ = stream_composition_calculator(tau[3], step[3], 'HPEnd')
    _, n_CO2_LR_LPEnd, _          = stream_composition_calculator(tau[4], step[4], 'LPEnd')
    n_tot_LR_HPEnd, n_CO2_LR_HPEnd, _ = stream_composition_calculator(tau[4], step[4], 'HPEnd')

    # Calculate purity, recovery, and mass balance 
    purity = (n_CO2_CnCDepres_HPEnd + (1 - beta) * n_CO2_LR_HPEnd) / \
             (n_tot_CnCDepres_HPEnd + (1 - beta) * n_tot_LR_HPEnd)
    recovery = (n_CO2_CnCDepres_HPEnd + (1 - beta) * n_CO2_LR_HPEnd) / \
               (n_CO2_CoCPres_HPEnd + n_CO2_ads_HPEnd)
    mass_balance = (n_CO2_CnCDepres_HPEnd + n_CO2_ads_LPEnd + n_CO2_HR_LPEnd + n_CO2_LR_HPEnd) / \
                   (n_CO2_CoCPres_HPEnd + n_CO2_ads_HPEnd + n_CO2_HR_HPEnd + n_CO2_LR_LPEnd)
    return purity, recovery, mass_balance


