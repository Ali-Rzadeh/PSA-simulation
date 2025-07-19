import numpy as np
print("Importing process_input_parameters...")  
def process_input_parameters(process_vars, material, N):
    print("Processing input parameters...")
    # --- Extract design variables
    L, P_0, ndot_0, t_ads, alpha, beta, P_I, P_l = process_vars

    # --- Material unpacking
    material_property = material[0]      # [rho_s, deltaU1, deltaU2]
    isotherm_par = material[1]           # 13-element array

    # --- Operating parameters
    t_pres      = 20
    t_CnCdepres = 30
    t_CoCdepres = 70
    t_LR        = t_ads
    t_HR        = t_LR
    tau         = 0.5
    P_inlet     = 1.02

    # --- Gas properties
    R      = 8.314
    T_0    = 313.15
    y_0    = 0.15
    Ctot_0 = P_0 / (R * T_0)
    v_0    = ndot_0 / Ctot_0
    mu     = 1.72e-5
    epsilon = 0.37
    D_m    = 1.2995e-5
    K_z    = 0.09
    C_pg   = 30.7
    C_pa   = 30.7
    MW_CO2 = 0.04402
    MW_N2  = 0.02802
    feed_gas = 'Constant Velocity'  # Could be changed later

    # --- Adsorbent properties
    ro_s      = material_property[0]
    r_p       = 1e-3
    C_ps      = 1070
    q_s       = 5.84
    q_s0      = q_s * ro_s
    k_CO2_LDF = 0.1631
    k_N2_LDF  = 0.2044
    deltaU    = [material_property[1], material_property[2]]

    # --- Isotherm parameters
    q_s_b     = [isotherm_par[0], isotherm_par[6]]
    q_s_d     = [isotherm_par[1], isotherm_par[7]]
    b         = [isotherm_par[2], isotherm_par[8]]
    d         = [isotherm_par[3], isotherm_par[9]]
    deltaU_b  = [isotherm_par[4], isotherm_par[10]]
    deltaU_d  = [isotherm_par[5], isotherm_par[11]]
    extra_iso = isotherm_par[12]

    # --- Param vector (size 39)
    Params = np.zeros(39)
    Params[0]  = N
    Params[1]  = deltaU[0]
    Params[2]  = deltaU[1]
    Params[3]  = ro_s
    Params[4]  = T_0
    Params[5]  = epsilon
    Params[6]  = r_p
    Params[7]  = mu
    Params[8]  = R
    Params[9]  = v_0
    Params[10] = q_s0
    Params[11] = C_pg
    Params[12] = C_pa
    Params[13] = C_ps
    Params[14] = D_m
    Params[15] = K_z
    Params[16] = P_0
    Params[17] = L
    Params[18] = MW_CO2
    Params[19] = MW_N2
    Params[20] = k_CO2_LDF
    Params[21] = k_N2_LDF
    Params[22] = y_0
    Params[23] = tau
    Params[24] = P_l
    Params[25] = P_inlet
    Params[26] = 1         # y_LR (to be updated during simulation)
    Params[27] = 1         # T_LR
    Params[28] = 1         # ndot_LR
    Params[29] = alpha
    Params[30] = beta
    Params[31] = P_I
    Params[32] = y_0       # y_HR
    Params[33] = T_0       # T_HR
    Params[34] = ndot_0 * beta
    Params[35] = 0.01      # y_CoCPressurization
    Params[36] = T_0
    Params[37] = ndot_0
    Params[38] = 1 if feed_gas.lower() == 'constant pressure' else 0

    # --- Step times [t_pres, t_ads, t_CnCdepres, t_LR, t_CoCdepres, t_HR]
    Times = np.array([t_pres, t_ads, t_CnCdepres, t_LR, t_CoCdepres, t_HR])

    # --- Isotherm param vector
    IsothermParams = np.array(q_s_b + q_s_d + b + d + deltaU_b + deltaU_d + [extra_iso])

    # --- Economic parameters
    desired_flow            = 100
    electricity_cost        = 0.07
    hours_per_year          = 8000
    life_equip              = 20
    life_adsorbent          = 5
    cepci                  = 536.4
    cycle_time              = t_pres + t_ads + t_HR + t_CnCdepres + t_LR

    EconomicParams = np.zeros(7)
    EconomicParams[0] = desired_flow
    EconomicParams[1] = electricity_cost
    EconomicParams[2] = cycle_time
    EconomicParams[3] = hours_per_year
    EconomicParams[4] = life_equip
    EconomicParams[5] = life_adsorbent
    EconomicParams[6] = cepci

    return Params, IsothermParams, Times, EconomicParams
