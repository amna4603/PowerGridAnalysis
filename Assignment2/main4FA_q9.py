#%%
# -*- coding: utf-8 -*-
"""
Main Script for Fault Analysis
"""
import numpy as np
import pandas as pd
import FaultAnalysis_46705 as fa    # import Fault Analysis functions
import LoadNetworkData4FA as lnd4fa # load the network data to global variables
filename = "./TestSystem4FA.txt"
lnd4fa.LoadNetworkData4FA(filename) # makes Zbus0 available as lndfa.Zbus0 etc.
# Carry out the fault analysis ... 

cur_vs_bus = []
for FaultBus in range(1,len(lnd4fa.bus_labels)+1): # loop through all busses
    # FaultType: 0 = 3-phase balanced fault; 1 = Single Line-to-Ground fault;
    #            2 = Line-to-Line fault;     3 = Double Line-to-Ground fault.
    FaultType = 0
    FaultImpedance = 0 # (in pu) 
    PrefaultVoltage = 1.000 # (in pu)
    # Iph: phase current array (0: phase a; 1: phase b; 2: phase c). 
    # Vph_mat: phase line-to-ground voltages (rows: busses; columns: phases a, b, c).
    Iph,Vph_mat = fa.FaultAnalysis(lnd4fa.Zbus0,lnd4fa.Zbus1,lnd4fa.Zbus2,lnd4fa.bus_to_ind, 
                                    FaultBus,FaultType,FaultImpedance,PrefaultVoltage)
    # Display results
    cur_vs_bus.append((FaultBus, abs(Iph[0])))
    # fa.DisplayFaultAnalysisResults(Iph,Vph_mat,FaultBus,FaultType,FaultImpedance,PrefaultVoltage)
print('**********End of Fault Analysis**********')
cur_vs_bus = np.array(cur_vs_bus)
df = pd.DataFrame(cur_vs_bus, columns=['Fault Bus', 'Fault Current (pu)'])
df['Fault Bus'] = df['Fault Bus'].astype(int)

Sbase = lnd4fa.MVA_base * 1e6 # Convert MVA to VA
Vbase_z1 = lnd4fa.bus_kv[0]

Ibase_array = []
for bus in df['Fault Bus']:
    Vbase = lnd4fa.bus_kv[bus-1] * 1000
    Ibase = Sbase / (Vbase * np.sqrt(3))
    Ibase_array.append(Ibase)
Ibase_array = np.array(Ibase_array)

df['Fault Current (kA)'] = df['Fault Current (pu)'] * Ibase_array / 1000
df['Fault Current (kA)'] = df['Fault Current (kA)'].round(2)
df.set_index('Fault Bus', inplace=True)
latex_code = df.to_latex()
print(latex_code)
# %%
