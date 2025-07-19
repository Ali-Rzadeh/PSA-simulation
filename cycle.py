print ("ssc")
print("This is cycle.py")
import numpy as np

from scipy.integrate import solve_ivp
from ProcessInput import process_input_parameters
from Isotherm import isotherm
import Jacobian as jac

# 
from Adsorbtion import func_adsorption

#from HelperFunctions import stream_composition_calculator


from CnCDepressurization import func_cnc_depressurization
from CoCPressurization import func_coc_pressurization
from HeavyReflux import func_heavy_reflux
from LightReflux import func_light_reflux
from HelperFunctions import stream_composition_calculator , velocity_correction , velocity_cleanup, process_evaluation

def unpack_params(Params):
    # I have not included the ndot_0 here, this must be checked.
    return (
        Params[3],             
        Params[4],              
        Params[5],             
        Params[6],             
        Params[7],             
        Params[8],             
        Params[9],             
        Params[10]/Params[3],             
        Params[16],             
        Params[17],             
        Params[18],             
        Params[19],             
        Params[22],             
        Params[24],
        Params[25],
        Params[29],
        Params[30],
        Params[32],
        Params[33],
        Params[34],
    )


def psa_cycle(vars, material, x0 = None, run_type='ProcessEvaluation', N=10, it_disp='no'):
    # Initialize outputs
    objectives = np.zeros(2)
    constraints = np.zeros(3)

    # Load all parameters
    Params, IsothermParams, Times, EconomicParams = process_input_parameters(vars, material, N)
    ro_s, T_0, epsilon, r_p, mu, R, v_0, q_s0, P_0, L, MW_CO2, MW_N2, y_0, P_l, P_inlet, alpha, beta, y_HR, T_HR, ndot_HR = unpack_params(Params)

    # Step durations (dimensionless)
    tau = {
        'coc_pres': Times[0] * v_0 / L,
        'ads': Times[1] * v_0 / L,
        'hr': Times[5] * v_0 / L,
        'cnc_depres': Times[2] * v_0 / L,
        'lr': Times[3] * v_0 / L
    }


    tau = {
        'coc_pres': 1,
        'ads': 1 ,
        'hr': 1,
        'cnc_depres':1,
        'lr': 1
    }
    # Initial condition setup

    # if x0 is None:
    #   x0 = initialize_state_vector(N, y_0, P_l, P_0, T_0, IsothermParams, q_s0)

    if x0 is None:
        q = isotherm(y_0, P_l, 298.15, IsothermParams)
        x0 = np.zeros(5*N + 10)
        x0[0:N+2] = P_l / P_0
        x0[N+2] = y_0
        x0[N+3:2*N+4] = y_0
        x0[2*N+4:3*N+6] = q[0] / q_s0
        x0[3*N+6:4*N+8] = q[1] / q_s0
        x0[4*N+8] = 1
        x0[4*N+9:5*N+10] = 298.15 / T_0

    
    ###########################################################################
    # Define RHS and Jacobian patterns 
    rhs = {
        'coc_pres': lambda t, x: func_coc_pressurization(t, x, Params, IsothermParams),
        'ads': lambda t, x: func_adsorption(t, x, Params, IsothermParams),
        'hr': lambda t, x: func_heavy_reflux(t, x, Params, IsothermParams),
        'cnc_depres': lambda t, x: func_cnc_depressurization(t, x, Params, IsothermParams),
        'lr': lambda t, x: func_light_reflux(t, x, Params, IsothermParams)
    }
    # ODE integration options with placeholder Jacobians
    opts = {
        'coc_pres': {'jac': jac.jac_pressurization(N)},
        'ads': {'jac': jac.jac_adsorption(N)},
        'hr': {'jac': jac.jac_adsorption(N)},
        'cnc_depres': {'jac': jac.jac_cnc_depressurization(N)},
        'lr': {'jac': jac.jac_light_reflux(N)},
    }

    # Storage arrays
    a_in, b_in, c_in, d_in, e_in = [], [], [], [], []
    a_fin, b_fin, c_fin, d_fin, e_fin = [], [], [], [], []

# x0: initial state vector for first step
    for i in range(700):
        print(f"Cycle iteration: {i+1}")    
        # --- 1. CoC Pressurization Step ---
        a_in.append(x0.copy())
        sol1 = solve_ivp(rhs['coc_pres'], [0, tau['coc_pres']], x0, method='Radau', **opts['coc_pres'], atol=1e-2, rtol=1e-2)
        a = sol1.y.T
        print(f"Cycle iteration: {i+1} - CoC Pressurization Step completed with {len(a)} time points.")

        # Output corrections (boundary cleanups)
        idx = np.where(a[:, 0] < a[:, 1])[0]
        a[idx, 0] = a[idx, 1]
        a[idx, N+2] = a[idx, N+3]
        a[idx, 4*N+8] = a[idx, 4*N+9]
        a[:, 2*N+4] = a[:, 2*N+5]
        a[:, 3*N+6] = a[:, 3*N+7]
        a[:, 3*N+5] = a[:, 3*N+4]
        a[:, 4*N+7] = a[:, 4*N+6]
        a[:, N+2:2*N+3] = np.clip(a[:, N+2:2*N+3], 0, 1)

        # Stream composition at both ends
        totalFront, CO2Front, _ = stream_composition_calculator(sol1.t * L / v_0, a, 'HPEnd')
        totalEnd, CO2End, _ = stream_composition_calculator(sol1.t * L / v_0, a, 'LPEnd')
        a_fin.append(np.concatenate([a[-1], [CO2Front, totalFront, CO2End, totalEnd]]))

        # Prepare initial state for Adsorption
        x10 = a[-1].copy() # take the last state (last time step) from CoC Pressurization
        x10[0] = P_inlet
        x10[N+1] = 1
        x10[N+2] = y_0
        x10[2*N+3] = x10[2*N+2]
        x10[4*N+8] = 1
        x10[5*N+9] = x10[5*N+8]
        b_in.append(x10.copy())

        statesIC = a[0, np.r_[1:N+1, N+3:2*N+3, 2*N+5:3*N+5, 3*N+7:4*N+7, 4*N+9:5*N+9]]

        # --- 2. Adsorption Step ---
        sol2 = solve_ivp(rhs['ads'], [0, tau['ads']], x10, method='Radau', **opts['ads'], atol=1e-2, rtol=1e-2)
        b = sol2.y.T
        print(f"Cycle iteration: {i+1} - Adsorption Step completed with {len(b)} time points.")

        idx = np.where(b[:, N] < 1)[0]
        b[idx, N+1] = b[idx, N]
        b[:, 2*N+4] = b[:, 2*N+5]
        b[:, 3*N+6] = b[:, 3*N+7]
        b[:, 3*N+5] = b[:, 3*N+4]
        b[:, 4*N+7] = b[:, 4*N+6]
        b[:, N+2:2*N+3] = np.clip(b[:, N+2:2*N+3], 0, 1)

        if Params[-1] == 0:
            b = velocity_cleanup(b)

        totalFront, CO2Front, _ = stream_composition_calculator(sol2.t * L / v_0, b, 'HPEnd')
        totalEnd, CO2End, TEnd = stream_composition_calculator(sol2.t * L / v_0, b, 'LPEnd')
        b_fin.append(np.concatenate([b[-1], [CO2Front, totalFront, CO2End, totalEnd]]))

        # Update params for Light Reflux step
        y_LR = CO2End / totalEnd
        T_LR = TEnd
        ndot_LR = totalEnd / tau['ads']
        Params[26] = y_LR
        Params[27] = T_LR
        Params[28] = ndot_LR

        # Prepare initial for Heavy Reflux
        x20 = b[-1].copy()
        x20[0] = P_inlet
        x20[0] = x20[1]
        x20[N+1] = 1
        x20[N+2] = y_HR
        x20[2*N+3] = x20[2*N+2]
        x20[4*N+8] = T_HR / T_0
        x20[5*N+9] = x20[5*N+8]
        c_in.append(x20.copy())

        # --- 3. Heavy Reflux Step ---
        sol3 = solve_ivp(rhs['hr'], [0, tau['hr']], x20, method='Radau', **opts['hr'], atol=1e-2, rtol=1e-2)
        c = sol3.y.T
        print(f"Cycle iteration: {i+1} - Heavy Reflux Step completed with {len(c)} time points.")   

        idx = np.where(c[:, N] < 1)[0]
        c[idx, N+1] = c[idx, N]
        c[:, 2*N+4] = c[:, 2*N+5]
        c[:, 3*N+6] = c[:, 3*N+7]
        c[:, 3*N+5] = c[:, 3*N+4]
        c[:, 4*N+7] = c[:, 4*N+6]
        c[:, N+2:2*N+3] = np.clip(c[:, N+2:2*N+3], 0, 1)

        if Params[-1] == 0:
            c = velocity_correction(c, ndot_HR, 'HPEnd')

        totalFront, CO2Front, _ = stream_composition_calculator(sol3.t * L / v_0, c, 'HPEnd')
        totalEnd, CO2End, _ = stream_composition_calculator(sol3.t * L / v_0, c, 'LPEnd')
        c_fin.append(np.concatenate([c[-1], [CO2Front, totalFront, CO2End, totalEnd]]))

        # Prepare initial for CnC Depressurization
        x30 = c[-1].copy()
        x30[0] = x30[1]
        x30[N+1] = x30[N]
        x30[N+2] = x30[N+3]
        x30[2*N+3] = x30[2*N+2]
        x30[4*N+8] = x30[4*N+9]
        x30[5*N+9] = x30[5*N+8]
        d_in.append(x30.copy())

        # --- 4. CnC Depressurization Step ---
        sol4 = solve_ivp(rhs['cnc_depres'], [0, tau['cnc_depres']], x30, method='Radau', **opts['cnc_depres'], atol=1e-2, rtol=1e-2)
        d = sol4.y.T
        print(f"Cycle iteration: {i+1} - CnC Depressurization Step completed with {len(d)} time points.")   

        idx = np.where(d[:, 1] < d[:, 0])[0]
        d[idx, 0] = d[idx, 1]
        d[:, 2*N+4] = d[:, 2*N+5]
        d[:, 3*N+6] = d[:, 3*N+7]
        d[:, 3*N+5] = d[:, 3*N+4]
        d[:, 4*N+7] = d[:, 4*N+6]
        d[:, N+2:2*N+3] = np.clip(d[:, N+2:2*N+3], 0, 1)

        totalFront, CO2Front, _ = stream_composition_calculator(sol4.t * L / v_0, d, 'HPEnd')
        totalEnd, CO2End, _ = stream_composition_calculator(sol4.t * L / v_0, d, 'LPEnd')
        d_fin.append(np.concatenate([d[-1], [CO2Front, totalFront, CO2End, totalEnd]]))

        # Prepare initial for Light Reflux
        x40 = d[-1].copy()
        x40[0] = P_l / P_0
        x40[N+2] = x40[N+3]
        x40[2*N+3] = y_LR
        x40[4*N+8] = x40[4*N+9]
        x40[5*N+9] = T_LR / T_0
        e_in.append(x40.copy())
        x40 = np.nan_to_num(x40, nan=1.0)
        # --- 5. Light Reflux Step ---
        sol5 = solve_ivp(rhs['lr'], [0, tau['lr']], x40, method='Radau', **opts['lr'], atol=1e-2, rtol=1e-2)
        e = sol5.y.T
        print(f"Cycle iteration: {i+1} - Light Reflux Step completed with {len(e)} time points.")   

        idx = np.where(e[:, 1] < e[:, 0])[0]
        e[idx, 0] = e[idx, 1]
        e[:, 2*N+4] = e[:, 2*N+5]
        e[:, 3*N+6] = e[:, 3*N+7]
        e[:, 3*N+5] = e[:, 3*N+4]
        e[:, 4*N+7] = e[:, 4*N+6]
        e[:, N+2:2*N+3] = np.clip(e[:, N+2:2*N+3], 0, 1)

        e = velocity_correction(e, ndot_LR * alpha, 'LPEnd')

        totalFront, CO2Front, TFront = stream_composition_calculator(sol5.t * L / v_0, e, 'HPEnd')
        totalEnd, CO2End, _ = stream_composition_calculator(sol5.t * L / v_0, e, 'LPEnd')
        e_fin.append(np.concatenate([e[-1], [CO2Front, totalFront, CO2End, totalEnd]]))

        # Prepare next cycle
        y_HR = CO2Front / totalFront
        T_HR = TFront
        ndot_HR = totalFront * beta / tau['hr']
        Params[32] = y_HR
        Params[33] = T_HR
        Params[34] = ndot_HR

        statesFC = e[-1, np.r_[1:N+1, N+3:2*N+3, 2*N+5:3*N+5, 3*N+7:4*N+7, 4*N+9:5*N+9]]

        # --- Cyclic Steady State Check ---
        CSS_states = np.linalg.norm(statesIC - statesFC)
        purity, recovery, massBalance = process_evaluation(a, b, c, d, e, sol1.t, sol2.t, sol3.t, sol4.t, sol5.t)
        print(f"Cycle iteration: {i+1} - CSS states: {CSS_states}, Purity: {purity}, Recovery: {recovery}, Mass Balance: {massBalance}")

        if CSS_states <= 1e-3 and abs(massBalance - 1) <= 0.005:
            break

        
        # x0 for next cycle:
        x0 = e[-1].copy()
        x0[0] = x0[1]
        x0[N+1] = x0[N]
        x0[N+2] = y_0
        x0[2*N+3] = x0[2*N+2]
        x0[4*N+8] = 1
        x0[5*N+9] = x0[5*N+8]




 

    # --- Process Evaluation
    if run_type.lower() == 'processevaluation':
        purity, recovery, MB = process_evaluation([sol1, sol2, sol3, sol4, sol5], tau, Params)
        objectives[:] = [-purity, -recovery / MB]
        constraints[1] = max(0, 0.9 - recovery / MB)
        constraints[2] = max(0, y_0 - purity)

    elif run_type.lower() == 'economicevaluation':
        productivity, energy, purity = economic_evaluation(sol1, sol2, sol3, sol4, sol5, tau, Params, EconomicParams)
        objectives[:] = [-productivity, energy]
        constraints[2] = max(0, 0.9 - purity)

    else:
        raise ValueError(f"Invalid run_type: {run_type}")

    return objectives, constraints, sol1.y, sol2.y, sol3.y, sol4.y, sol5.y, sol1.t, sol2.t, sol3.t, sol4.t, sol5.t
