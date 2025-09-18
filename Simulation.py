from cycle import psa_cycle
import numpy as np
import matplotlib.pyplot as plt

import time


start_time = time.time()

type = 'ProcessEvaluation'

#UTSA-16
#material_property = [1659, -33000, -12000]
#IsothermParams = [5,3,9.46e-11,6.15e-16,-33000,-48000,12.7,0,4.29e-10,0,-12300,0,0] 

#Zolite13X
material_property = [1.130000e+03, -3.600000e+04, -1.580000e+04]
IsothermParams = [3.090000e+00, 2.540000e+00, 8.650000e-07, 2.630000e-08, -3.664121e+04 ,-3.569066e+04, 5.840000e+00 ,0.000000e+00 ,2.500000e-06, 0.000000e+00, -1.580000e+04, 0.000000e+00, 1.000000e+00] 


material = [material_property, IsothermParams]
#material = [SimParam[11, :], IsothermPar[11, :]]  
N = 30
type = 'ProcessEvaluation'


    # Note: Replace these with your fixed values if not optimizing
#process_variables = [
 #   1.0,                      #  Bed length or fixed var
 #   1.64e5,  #1.42e5,                     # Column pressure [Pa]
 #   18.897,  #  Utsa = 19.08947,  # Inlet molar flux 
 #   486.53,   #828.3232,                     # Adsorption time [s]
 #   0.1,    #0.11,                     # Light reflux ratio
 #   1.0,                     # Heavy reflux ratio
 #   1.0e4,                      #  Intermediate pressure [Pa]
 #   1.0e4                       # Purge pressure [Pa]
#]

opt_row = [7.63336e+05, 4.60896e+01, 9.51027e-02, 9.65920e-01, 1.0, 1.04979e+04, 2.94]

process_variables = [
    1.0,
    float(opt_row[0]),
    float(opt_row[0] * opt_row[3] / 8.314 / 313.15),
    float(opt_row[1]),
    float(opt_row[2]),
    float(opt_row[4]),
    1.0e4,
    float(opt_row[5]),                   # Purge pressure [Pa]
]

#process_input_parameters(process_variables, material, N)
#try:
purity, recovery, productivity, energy_requirement,b,c,d,e = psa_cycle(process_variables, material, None, type, N)

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
print("Purity:", purity)
print("Recovery:", recovery)
print("Productivity:", productivity)
print("Energy Requirement:", energy_requirement)    
#print("b:", b)
b2= b[2*N+4:3*N+6,-1]
c2= c[2*N+4:3*N+6,-1]
b3= b[2*N+4:3*N+6,0]
c3= c[2*N+4:3*N+6,0]


d4= d[2*N+4:3*N+6,-1]
e4= e[2*N+4:3*N+6,-1]
d5= d[2*N+4:3*N+6,0]
e5= e[2*N+4:3*N+6,0]
concat = np.concatenate((b2,c2), axis=0)
plt.plot(concat,label='CO2 concentration at adsorptiom')
plt.title('CO2 concentration at adsorption step time = end')
plt.savefig("1solid.png")   # saves as PNG in current directory

plt.show()


concat2 = np.concatenate((b3,c3), axis=0)
plt.plot(concat2,label='CO2 concentration at adsorptiom')
plt.title('CO2 concentration at adsorption step time = 0')
plt.savefig("2solid.png")   # saves as PNG in current directory

plt.show()

concat3 = np.concatenate((d4,e4), axis=0)
plt.plot(concat3,label='CO2 concentration at adsorptiom')
plt.title('CO2 concentration at desorptiom step time = end')
plt.savefig("3solid.png")   # saves as PNG in current directory

plt.show()


concat4 = np.concatenate((d5,e5), axis=0)
plt.plot(concat4,label='CO2 concentration at adsorptiom')
plt.title('CO2 concentration at desorption step time = 0')
plt.savefig("4solid.png")   # saves as PNG in current directory

plt.show()









b2= b[N+2:2*N+4,-1]
c2= c[N+2:2*N+4,-1]
b3= b[N+2:2*N+4,0]
c3= c[N+2:2*N+4,0]


d4= d[N+2:2*N+4,-1]
e4= e[N+2:2*N+4,-1]
d5= d[N+2:2*N+4,0]
e5= e[N+2:2*N+4,0]
concat = np.concatenate((b2,c2), axis=0)
plt.plot(concat,label='CO2 concentration at adsorptiom')
plt.title('gas mole fraction at adsorption step time = end')
plt.savefig("1mole.png")   # saves as PNG in current directory

plt.show()


concat2 = np.concatenate((b3,c3), axis=0)
plt.plot(concat2,label='CO2 concentration at adsorptiom')
plt.title('gas mole fraction at adsorption step time = 0')
plt.savefig("2mole.png")   # saves as PNG in current directory

plt.show()

concat3 = np.concatenate((d4,e4), axis=0)
plt.plot(concat3,label='CO2 concentration at adsorptiom')
plt.title('gas mole fraction at desorptiom step time = end')
plt.savefig("3mole.png")   # saves as PNG in current directory

plt.show()


concat4 = np.concatenate((d5,e5), axis=0)
plt.plot(concat4,label='CO2 concentration at adsorptiom')
plt.title('gas mole fraction at desorption step time = 0')
plt.savefig("4mole.png")   # saves as PNG in current directory

plt.show()