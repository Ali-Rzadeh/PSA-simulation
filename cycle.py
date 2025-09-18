import numpy as np

from scipy.integrate import solve_ivp
from ProcessInput import process_input_parameters
from Isotherm import isotherm
import Jacobian as jac

# 
from Adsorbtion import func_adsorption
from CnCDepressurization import func_cnc_depressurization
from CoCPressurization import func_coc_pressurization
from HeavyReflux import func_heavy_reflux
from LightReflux import func_light_reflux
from HelperFunctions import stream_composition_calculator , velocity_correction , velocity_cleanup, process_evaluation, stream_composition_calculator_julia , CompressionEnergy, VacuumEnergy    



def psa_cycle(vars, material, x0 = None, run_type='ProcessEvaluation', N=10, it_disp='no'):
    # Initialize outputs
    objectives = np.zeros(2)
    constraints = np.zeros(3)

    # Load all parameters
    Params, IsothermParams, Times, EconomicParams = process_input_parameters(vars, material, N)


    N = int(Params[0])
    ro_s = Params[3]
    T_0 = Params[4]
    epsilon = Params[5]
    r_p = Params[6]
    mu = Params[7]
    R = Params[8]
    v_0 = Params[9]
    q_s0 = Params[10] / ro_s
    P_0 = Params[16]
    L = Params[17]
    MW_CO2 = Params[18]
    MW_N2 = Params[19]
    y_0 = Params[22]
    ndot_0 = vars[2]
    P_l = Params[24]
    P_inlet = Params[25]
    alpha = Params[29]
    beta = Params[30]
    y_HR = Params[32]
    T_HR = Params[33]
    ndot_HR = Params[34]

    
    # Step durations (dimensionless)
   
    t_coc_pres =Times[0]
    t_ads= Times[1] 
    t_hr =  Times[5] 
    t_cnc_depres = Times[2] 
    t_lr = Times[3]
  

    #dimensionless time
    tau_coc_pres = t_coc_pres * v_0 / L
    tau_ads = t_ads * v_0 / L
    tau_hr = t_hr * v_0 / L
    tau_cnc_depres = t_cnc_depres * v_0 / L
    tau_lr = t_lr * v_0 / L
    
    # Initial condition setup

    if x0 is None:
        q = isotherm(y_0, P_l, 298.15, IsothermParams)
        x0 = np.zeros(5*N + 10)
        x0[0:N+2] = P_l / P_0
        x0[N+2] = y_0
        x0[N+3:2*N+4] = y_0
        x0[2*N+4:3*N+6] = q[0] / q_s0
        x0[3*N+6:4*N+8] = q[1] / q_s0
        x0[4*N+8] = 1
        x0[4*N+9:5*N+10] = 298.15 / T_0  # this is also 1

    
    ###########################################################################
    # Define RHS functions for each step
    def rhs_coc_pres(t, x):
        return func_coc_pressurization(t, x, Params, IsothermParams)
    def rhs_ads(t, x):
        return func_adsorption(t, x, Params, IsothermParams)  
    def rhs_hr(t, x):
        return func_heavy_reflux(t, x, Params, IsothermParams)  
    def rhs_cnc_depres(t, x):
        return func_cnc_depressurization(t, x, Params, IsothermParams)
    def rhs_lr(t, x):
        return func_light_reflux(t, x, Params, IsothermParams)

    ###########################################################################
    # Storage arrays
  #  a_in, b_in, c_in, d_in, e_in = [], [], [], [], []
    a_fin, b_fin, c_fin, d_fin, e_fin = [], [], [], [], []

# x0: initial state vector for first step
    for i in range(700):
       # print(f"Cycle iteration: {i+1}")
       # if i == 0:
           #  statesIC =x0[0, np.r_[1:N+1, N+3:2*N+3, 2*N+5:3*N+5, 3*N+7:4*N+7, 4*N+9:5*N+9]]
        # --- 1. CoC Pressurization Step ---
       # a_in.append(x0.copy())

        sol1 = solve_ivp( rhs_coc_pres, [0, tau_coc_pres], x0, method='BDF', jac_sparsity = jac.jac_pressurization(N),atol=1e-6, rtol=1e-6 )
        a = sol1.y.T
        #print(f"Cycle iteration: {i+1} - CoC Pressurization Step completed with {len(a)} time points.")
       # print("nfev:", sol1.nfev, "njev:", sol1.njev, "nlu:", sol1.nlu)
        # Output corrections (boundary cleanups)
        idx = np.where(a[:, 0] < a[:, 1])[0]
        a[idx, 0] = a[idx, 1]
        a[idx, N+2] = a[idx, N+3]
        a[idx, 4*N+8] = a[idx, 4*N+9]
        a[:, 2*N+4] = a[:, 2*N+5]
        a[:, 3*N+6] = a[:, 3*N+7]
        a[:, 3*N+5] = a[:, 3*N+4]
        a[:, 4*N+7] = a[:, 4*N+6]
        a[:, N+2:2*N+4] = np.clip(a[:, N+2:2*N+4], 0, 1)

        # Stream composition at both ends
        totalFront, CO2Front, _ = stream_composition_calculator(sol1.t * L / v_0, a,Params, 'HPEnd')
        totalEnd, CO2End, _ = stream_composition_calculator(sol1.t * L / v_0, a, Params, 'LPEnd')
        a_fin.append(np.concatenate([a[-1], [CO2Front, totalFront, CO2End, totalEnd]]))  #store the last state of the step  

        # Prepare initial state for Adsorption
        x10 = a[-1].copy() # take the last state (last time step) from CoC Pressurization


        x10[0] = P_inlet
        x10[N+1] = 1
        x10[N+2] = y_0
        x10[2*N+3] = x10[2*N+2]
        x10[4*N+8] = 1
        x10[5*N+9] = x10[5*N+8]
      #  b_in.append(x10.copy())

        statesIC = a[0, np.r_[1:N+1, N+3:2*N+3, 2*N+5:3*N+5, 3*N+7:4*N+7, 4*N+9:5*N+9]]


        # --- 2. Adsorption Step ---                                            
        sol2 = solve_ivp(rhs_ads, [0, tau_ads], x10, method='BDF', jac_sparsity = jac.jac_adsorption(N),   atol=1e-6, rtol=1e-6)                                            
        b = sol2.y.T                                            
      #  print(f"Cycle iteration: {i+1} - Adsorption Step completed with {len(b)} time points.")                                            
                                             
        idx = np.where(b[:, N] < 1)[0]                                            
        b[idx, N+1] = b[idx, N]                                            
        b[:, 2*N+4] = b[:, 2*N+5]                                            
        b[:, 3*N+6] = b[:, 3*N+7]
        b[:, 3*N+5] = b[:, 3*N+4]
        b[:, 4*N+7] = b[:, 4*N+6]
        b[:, N+2:2*N+4] = np.clip(b[:, N+2:2*N+4], 0, 1)

        if Params[-1] == 0:
            b = velocity_cleanup(b, Params)

        totalFront, CO2Front, _ = stream_composition_calculator(sol2.t * L / v_0, b, Params, 'HPEnd')
        totalEnd, CO2End, TEnd = stream_composition_calculator(sol2.t * L / v_0, b, Params, 'LPEnd')
        b_fin.append(np.concatenate([b[-1], [CO2Front, totalFront, CO2End, totalEnd]]))

        # Update params for Light Reflux step
        y_LR = CO2End / totalEnd
        T_LR = TEnd
        ndot_LR = totalEnd / t_ads
        Params[26] = y_LR
        Params[27] = T_LR
        Params[28] = ndot_LR

        # Prepare initial for Heavy Reflux
        x20 = b[-1].copy()
       # x20[0] = P_inlet
        x20[0] = x20[1]
        x20[N+1] = 1.0
        x20[N+2] = y_HR
        x20[2*N+3] = x20[2*N+2]
        x20[4*N+8] = T_HR / T_0
        x20[5*N+9] = x20[5*N+8]
        #c_in.append(x20.copy())

        # --- 3. Heavy Reflux Step ---
        sol3 = solve_ivp(rhs_hr, [0, tau_hr], x20, method='BDF', jac_sparsity = jac.jac_adsorption(N),  atol=1e-6, rtol=1e-6)
        c = sol3.y.T
       # print(f"Cycle iteration: {i+1} - Heavy Reflux Step completed with {len(c)} time points.")   

        idx = np.where(c[:, N] < 1)[0]
        c[idx, N+1] = c[idx, N]
        c[:, 2*N+4] = c[:, 2*N+5]
        c[:, 3*N+6] = c[:, 3*N+7]
        c[:, 3*N+5] = c[:, 3*N+4]
        c[:, 4*N+7] = c[:, 4*N+6]
        c[:, N+2:2*N+4] = np.clip(c[:, N+2:2*N+4], 0, 1)

        if Params[-1] == 0:
            c = velocity_correction(c, ndot_HR, Params, 'HPEnd')

        totalFront, CO2Front, _ = stream_composition_calculator(sol3.t * L / v_0, c, Params, 'HPEnd')
        totalEnd, CO2End, _ = stream_composition_calculator(sol3.t * L / v_0, c, Params, 'LPEnd')
        c_fin.append(np.concatenate([c[-1], [CO2Front, totalFront, CO2End, totalEnd]]))

        # Prepare initial for CnC Depressurization
        x30 = c[-1].copy()
        x30[0] = x30[1]
        x30[N+1] = x30[N]
        x30[N+2] = x30[N+3]
        x30[2*N+3] = x30[2*N+2]
        x30[4*N+8] = x30[4*N+9]
        x30[5*N+9] = x30[5*N+8]
       # d_in.append(x30.copy())

        # --- 4. CnC Depressurization Step ---
        sol4 = solve_ivp(rhs_cnc_depres, [0, tau_cnc_depres], x30, method='BDF', jac_sparsity = jac.jac_cnc_depressurization(N),  atol=1e-6, rtol=1e-6)
        d = sol4.y.T
      #  print(f"Cycle iteration: {i+1} - CnC Depressurization Step completed with {len(d)} time points.")   

        idx = np.where(d[:, 1] < d[:, 0])[0]
        d[idx, 0] = d[idx, 1]
        d[:, 2*N+4] = d[:, 2*N+5]
        d[:, 3*N+6] = d[:, 3*N+7]
        d[:, 3*N+5] = d[:, 3*N+4]
        d[:, 4*N+7] = d[:, 4*N+6]
        d[:, N+2:2*N+4] = np.clip(d[:, N+2:2*N+4], 0, 1)

        totalFront, CO2Front, _ = stream_composition_calculator(sol4.t * L / v_0, d, Params, 'HPEnd')
        totalEnd, CO2End, _ = stream_composition_calculator(sol4.t * L / v_0, d, Params, 'LPEnd')
        d_fin.append(np.concatenate([d[-1], [CO2Front, totalFront, CO2End, totalEnd]]))

        # Prepare initial for Light Reflux
        x40 = d[-1].copy()
        x40[0] = P_l / P_0
        x40[N+2] = x40[N+3]
        x40[2*N+3] = y_LR
        x40[4*N+8] = x40[4*N+9]
        x40[5*N+9] = T_LR / T_0
       # e_in.append(x40.copy())
       # x40 = np.nan_to_num(x40, nan=1.0)
        # --- 5. Light Reflux Step ---
        sol5 = solve_ivp(rhs_lr, [0, tau_lr], x40, method='BDF', jac_sparsity = jac.jac_light_reflux(N),  atol=1e-6, rtol=1e-6)
        e = sol5.y.T
        #print(f"Cycle iteration: {i+1} - Light Reflux Step completed with {len(e)} time points.")   

        idx = np.where(e[:, 1] < e[:, 0])[0]
        e[idx, 0] = e[idx, 1]
        e[:, 2*N+4] = e[:, 2*N+5]
        e[:, 3*N+6] = e[:, 3*N+7]
        e[:, 3*N+5] = e[:, 3*N+4]
        e[:, 4*N+7] = e[:, 4*N+6]
        e[:, N+2:2*N+4] = np.clip(e[:, N+2:2*N+4], 0, 1)

        e = velocity_correction(e, ndot_LR * alpha, Params, 'LPEnd')

        totalFront, CO2Front, TFront = stream_composition_calculator(sol5.t * L / v_0, e, Params, 'HPEnd')
        totalEnd, CO2End, _ = stream_composition_calculator(sol5.t * L / v_0, e, Params, 'LPEnd')
        e_fin.append(np.concatenate([e[-1], [CO2Front, totalFront, CO2End, totalEnd]]))

        # Prepare next cycle
        y_HR = CO2Front / totalFront
        T_HR = TFront
        ndot_HR = totalFront * beta / t_hr
        Params[32] = y_HR
        Params[33] = T_HR
        Params[34] = ndot_HR


        x0 = e[-1].copy()
        x0[0] = x0[1]
        x0[N+1] = x0[N]
        x0[N+2] = y_0
        x0[2*N+3] = x0[2*N+2]
        x0[4*N+8] = 1
        x0[5*N+9] = x0[5*N+8]

        statesFC = e[-1, np.r_[1:N+1, N+3:2*N+3, 2*N+5:3*N+5, 3*N+7:4*N+7, 4*N+9:5*N+9]]

        # --- Cyclic Steady State Check ---
        
        CSS_states = np.linalg.norm(statesIC - statesFC)
        purity, recovery, massBalance = process_evaluation(a, b, c, d, e, sol1.t, sol2.t, sol3.t
                                                            , sol4.t, sol5.t, Params)
       # print(f"Cycle iteration: {i+1} - CSS states: {CSS_states}, Purity: {purity}, Recovery: {recovery}, Mass Balance: {massBalance}")

        if CSS_states <= 1e-3 and abs(massBalance - 1) <= 0.005:
            print(f"Cyclic steady state achieved after {i+1} cycles.")  
            print(f"Final CSS states: {CSS_states}, Mass Balance: {massBalance}") 
            break

       # statesIC = statesFC
   # end of main cycle iterations must be here

   # Post CSS analysis (Process and Economic Evaluation)
    purity, recovery, MB = process_evaluation(a, b, c, d, e, sol1.t, sol2.t,sol3.t, sol4.t, sol5.t, Params)
    desired_flow = EconomicParams[0]
    cycletime = t_cnc_depres + t_ads+ t_hr + t_coc_pres + t_lr    
    ntot_pres, _, _ = stream_composition_calculator_julia(sol1.t * L / v_0, a, Params, 'HPEnd')
    ntot_ads, _, _ = stream_composition_calculator_julia(sol2.t * L / v_0, b, Params, 'HPEnd')
    gas_fed = (ntot_pres + ntot_ads)

    r_in = ((desired_flow *cycletime/ gas_fed)/3.14159)**0.5

    E_pre = CompressionEnergy(sol1.t * L / v_0, a, r_in, Params,1e5)
    E_feed = CompressionEnergy(sol2.t * L / v_0, b,r_in ,Params, 1e5)
    E_hr = CompressionEnergy(sol3.t * L / v_0, c,r_in, Params, 1e5)
    
    E_blow = VacuumEnergy(sol4.t * L / v_0, d,r_in, "HPEnd", Params, 1e5)
    E_evac = VacuumEnergy(sol5.t * L / v_0, e,r_in,"EPEnd" ,Params, 1e5)
    energy_per_cycle = (E_pre + E_feed + E_hr + E_blow + E_evac) 


    _, nCO2_CnCD, _ = stream_composition_calculator_julia(sol4.t * L / v_0, d, Params, 'HPEnd')
    _, nCO2_LR, _ = stream_composition_calculator_julia(sol5.t * L / v_0, e, Params, 'HPEnd')
    CO2_recovered_cycle = (nCO2_CnCD +(1- Params[30])* nCO2_LR)*r_in **2*3.14159* MW_CO2/1e3
    CO2_recovered_cycle2 = (nCO2_CnCD +(1- Params[30])* nCO2_LR)*r_in **2*3.14159

    mass_adsorbent = L * 3.14159 * r_in**2 * ro_s * (1 - epsilon)
    productivity = CO2_recovered_cycle2 / (mass_adsorbent * cycletime)
    energy_requirement = energy_per_cycle / CO2_recovered_cycle

   # con = recovery/MB -0.9
   # if con<0:
   #     constraints[1] = -con



    return purity, recovery, productivity, energy_requirement
