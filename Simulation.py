from cycle import psa_cycle
import pandas as pd
from ProcessInput import process_input_parameters

# Read CSV files 
isotherm_df = pd.read_csv(r'C:\Users\Ali Rahimzadeh\OneDrive\Desktop\PSA\Isotherm.csv')
simpar_df = pd.read_csv(r'C:\Users\Ali Rahimzadeh\OneDrive\Desktop\PSA\Sim par.csv')



IsothermPar = isotherm_df.to_numpy()
SimParam = simpar_df.to_numpy()

#print("IsothermPar:", IsothermPar)


N = 10
type = 'ProcessEvaluation'

for i in [11]:  
    IsothermParams = IsothermPar[i, :]
    material_property = SimParam[i, :]

material = [material_property, IsothermParams]
#material = [SimParam[11, :], IsothermPar[11, :]]  
N = 10
type = 'ProcessEvaluation'

#def PSACycleSimulation(x, material, type_, N):

    # material [1, 2] = [SimParam, IsothermPar]
    # process_variables = [length, pressure, inlet_flux, adsorption_time, light_reflux_ratio, heavy_reflux_ratio, intermediate_pressure, purge_pressure]


    # Note: Replace these with your fixed values if not optimizing
#process_variables = [
    #1.0,                      #  Bed length or fixed var
   # x[0],                     # Column pressure [Pa]
   # x[0] * x[3] / 8.314 / 313.15,  # Inlet molar flux 
   # x[1],                     # Adsorption time [s]
  #  x[2],                     # Light reflux ratio
 #   x[4],                     # Heavy reflux ratio
 #   1e4,                      #  Intermediate pressure [Pa]
 #   x[5]                      # Purge pressure [Pa]
#]

    # Note: Replace these with your fixed values if not optimizing
process_variables = [
    1.77,                      #  Bed length or fixed var
    1.42,                     # Column pressure [Pa]
    0.35,  # Inlet molar flux 
    828.32,                     # Adsorption time [s]
    0.11,                     # Light reflux ratio
    1.0,                     # Heavy reflux ratio
    0.7,                      #  Intermediate pressure [Pa]
    0.2                       # Purge pressure [Pa]
]

process_input_parameters(process_variables, material, N)
#try:
objectives, constraints = psa_cycle(process_variables, material, None, type, N)
#except Exception as e:
   # objectives = [1e5, 1e5]
    #constraints = [1, 1, 1]
#return objectives, constraints
print("Objectives:", objectives)
print("Constraints:", constraints)