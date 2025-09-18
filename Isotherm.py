import numpy as np
from ProcessInput import process_input_parameters




def isotherm(y, P, T, isotherm_parameters):
    """
    Compute molar loadings of a two-component mixture using a dual-site Langmuir isotherm.
    
    Parameters:
    - y: mole fraction of component 1 (float or ndarray)
    - P: total pressure [Pa]
    - T: temperature [K]
    - isotherm_parameters: list or array of 13 values
    
    Returns:
    - q: [q1, q2] molar loadings of components 1 and 2
    """
    R = 8.314
    
    #isotherm_parameters = [3.090000e+00, 2.540000e+00, 8.650000e-07, 2.630000e-08, -3.664121e+04 ,-3.569066e+04, 5.840000e+00 ,0.000000e+00 ,2.500000e-06, 0.000000e+00, -1.580000e+04, 0.000000e+00, 1.000000e+00] 

    # Unpack isotherm parameters
    q_s_b_1 = isotherm_parameters[0]
    q_s_d_1 = isotherm_parameters[2]
    q_s_b_2 = isotherm_parameters[1]
    q_s_d_2 = isotherm_parameters[3]
    
    b_1 = isotherm_parameters[4]
    d_1 = isotherm_parameters[6]
    b_2 = isotherm_parameters[5]
    d_2 = isotherm_parameters[7]
    
    deltaU_b_1 = isotherm_parameters[8]
    deltaU_d_1 = isotherm_parameters[10]
    deltaU_b_2 = isotherm_parameters[9]
    deltaU_d_2 = isotherm_parameters[11]
    
    input_mode = isotherm_parameters[12]

    # Temperature-dependent affinity constants
    B_1 = b_1 * np.exp(-deltaU_b_1 / (R * T))
    D_1 = d_1 * np.exp(-deltaU_d_1 / (R * T))
    B_2 = b_2 * np.exp(-deltaU_b_2 / (R * T))
    D_2 = d_2 * np.exp(-deltaU_d_2 / (R * T))

    # Input variables
    if input_mode == 0:
        P_1 = y * P
        P_2 = (1 - y) * P
        input_1 = P_1
        input_2 = P_2
    elif input_mode == 1:
        C_1 = y * P / R / T
        C_2 = (1 - y) * P / R / T
        input_1 = C_1
        input_2 = C_2
    else:
        raise ValueError("Specify whether the isotherm is in terms of concentration or partial pressure (0 or 1)")

    # Langmuir isotherm components
    q1_b = q_s_b_1 * B_1 * input_1 / (1 + B_1 * input_1 + B_2 * input_2)
    q1_d = q_s_d_1 * D_1 * input_1 / (1 + D_1 * input_1 + D_2 * input_2)
    q1 = q1_b + q1_d

    q2_b = q_s_b_2 * B_2 * input_2 / (1 + B_1 * input_1 + B_2 * input_2)
    q2_d = q_s_d_2 * D_2 * input_2 / (1 + D_1 * input_1 + D_2 * input_2)
    q2 = q2_b + q2_d

    q = np.array([q1, q2])

    return q
