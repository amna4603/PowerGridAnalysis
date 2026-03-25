"""
46705 - Power Grid Analysis
This file contains the definitions of the functions needed to
carry out Fault Analysis calculations in python.
"""

import numpy as np

# 1. the FaultAnalysis() function
def FaultAnalysis(Zbus0,Zbus1,Zbus2,bus_to_ind,fault_bus,fault_type,Zf,Vf):
   
    # calculate sequence fault currents
    Iseq = Calculate_Sequence_Fault_Currents(Zbus0,Zbus1,Zbus2,bus_to_ind,fault_bus,fault_type,Zf,Vf)
    # calculate sequence fault voltages
    Vseq_mat = Calculate_Sequence_Fault_Voltages(Zbus0,Zbus1,Zbus2,bus_to_ind,fault_bus,Vf,Iseq)
    # convert sequence currents to phase (fault) currents
    Iph = Convert_Sequence2Phase_Currents(Iseq)
    # convert sequence voltages to phase line-to-ground (fault) voltages
    Vph_mat = Convert_Sequence2Phase_Voltages(Vseq_mat)    
    return Iph, Vph_mat

# 1.1. the Calculate_Sequence_Fault_Currents() function
def Calculate_Sequence_Fault_Currents(Zbus0,Zbus1,Zbus2,bus_to_ind,fault_bus,fault_type,Zf,Vf):
#  fault_type: 0 = 3-phase balanced fault; 1 = Single Line-to-Ground fault;
#              2 = Line-to-Line fault;     3 = Double Line-to-Ground fault.    
    # Iseq current array: 
    # Iseq[0] = zero-sequence; Iseq[1] = positive-sequence; Iseq[2] = negative-sequence
    Iseq = np.zeros(3,dtype=complex)
    fb = bus_to_ind[fault_bus]
    if fault_type == 0:
        Iseq[0] = Iseq[2] = 0
        Iseq[1] = Vf/Zbus1[fb,fb]
    elif fault_type == 1:
        Iseq[0] = Iseq[1] = Iseq[2] = Vf/(Zbus0[fb,fb] + Zbus1[fb,fb] + Zbus2[fb,fb] + 3*Zf)
    elif fault_type == 2:
        Iseq[0] = 0
        Iseq[1] = Vf/(Zbus1[fb,fb] + Zbus2[fb, fb] + Zf)
        Iseq[2] = -Iseq[1]
    elif fault_type == 3:
        Zeq = Zbus2[fb,fb]*(Zbus0[fb,fb] + 3*Zf)/(Zbus2[fb,fb] + Zbus0[fb,fb] + 3*Zf)
        Iseq[1] = Vf/(Zbus1[fb,fb] + Zeq)
        Iseq[2] = -Iseq[1]*(Zbus0[fb,fb] + 3*Zf)/(Zbus0[fb,fb] + Zbus2[fb,fb] + 3*Zf)
        Iseq[0] = -Iseq[1]*Zbus2[fb,fb]/(Zbus0[fb,fb] + Zbus2[fb,fb] + 3*Zf)
    else:
        print('Unknown Fault Type')
    return Iseq

# 1.2 the Calculate_Sequence_Fault_Voltages() function
def Calculate_Sequence_Fault_Voltages(Zbus0,Zbus1,Zbus2,bus_to_ind,fault_bus,Vf,Iseq):
    fb = bus_to_ind[fault_bus]
    total_bus_number = Zbus0.shape[0]
    temp_list = []
    for i in range(total_bus_number):
        bus_index = bus_to_ind[i+1]
        V1 = Vf - Zbus1[bus_index,fb]*Iseq[1]
        V2 = -Zbus2[bus_index,fb]*Iseq[2]
        V0 = -Zbus0[bus_index,fb]*Iseq[0]
        temp_list.append([V0,V1,V2])
    Vseq_mat = np.array(temp_list, dtype = complex)
    return Vseq_mat

# 1.3. the Convert_Sequence2Phase_Currents() function
def Convert_Sequence2Phase_Currents(Iseq):
    theta = np.deg2rad(120)
    a = np.exp(1j*theta)
    Ir = Iseq[0] + Iseq[1] + Iseq[2]
    Is = Iseq[0] + a**2*Iseq[1] + a*Iseq[2]
    It = Iseq[0] + a*Iseq[1] + a**2*Iseq[2]
    Iph = np.array([Ir,Is,It], dtype = complex)
    return Iph

# 1.4 the Convert_Sequence2Phase_Voltages() function
def Convert_Sequence2Phase_Voltages(Vseq_mat):
    theta = np.deg2rad(120)
    a = np.exp(1j*theta)
    temp_list = []
    for V_seq in Vseq_mat:
        VR = V_seq[0] + V_seq[1] + V_seq[2]
        VS = V_seq[0] + a**2 * V_seq[1] + a*V_seq[2]
        VT = V_seq[0] + a*V_seq[1] + a**2*V_seq[2]
        temp_list.append([VR, VS, VT])
    Vph_mat = np.array(temp_list, dtype = complex)
    return Vph_mat

# ####################################################
# #  Displaying the results in the terminal window   #
# ####################################################
# 2. the DisplayFaultAnalysisResults() function
def DisplayFaultAnalysisResults(Iph,Vph_mat,fault_bus,fault_type,Zf,Vf):
    fault_types = ['3-phase balanced fault',
                   'Single Line-to-Ground fault',
                   'Line-to-Line fault',
                   'Double Line-to-Ground fault'
                   ]
    fault_phases = ''
    if Iph[0] != 0:
        fault_phases += 'a'
    if Iph[1] != 0:
        fault_phases += 'b'
    if Iph[2] != 0:
        fault_phases += 'c'

    print('=================================================================')
    print('|                    Fault Analysis Results                     |')
    print('=================================================================')
    print(f'| {fault_types[fault_type]} at Bus {fault_bus}, phase {fault_phases}                 |')
    print(f'| Prefault Voltage: {Vf:.3f} (pu)                                  |')
    print(f'| Fault Impedance: {Zf:.3f} (pu)                                   |')
    print('=================================================================')  
    print('| Phase Currents                                                |')
    print('|            phase a    |        phase b    |      phase c      |')
    print('|     ------------------|-------------------|-------------------|')
    print(f'|     Mag (pu) Ang (deg)| Mag (pu) Ang (deg)| Mag (pu) Ang (deg)|')
    print(f'|       {abs(Iph[0]):.3f}     {np.rad2deg(np.angle(Iph[0])):.3f} |   {abs(Iph[1]):.3f}     {np.rad2deg(np.angle(Iph[1])):.3f} |   {abs(Iph[2]):.3f}     {np.rad2deg(np.angle(Iph[2])):.3f} |')
    print('|                                                               |')
    print('=================================================================')  
    print('| Phase Line-to-Ground Voltages                                 |')
    print('=================================================================')  
    print('|            phase a    |        phase b    |      phase c      |')
    print('|Bus| ------------------|-------------------|-------------------|')
    print(f'|   | Mag (pu) Ang (deg)| Mag (pu) Ang (deg)| Mag (pu) Ang (deg)|')
    print('|---|-------------------|-------------------|-------------------|')
    for i in range(Vph_mat.shape[0]):
        print(f'| {i+1} |   {abs(Vph_mat[i,0]):.3f}     {np.rad2deg(np.angle(Vph_mat[i,0])):.3f} |     {abs(Vph_mat[i,1]):.3f}   {np.rad2deg(np.angle(Vph_mat[i,1])):.3f} |     {abs(Vph_mat[i,2]):.3f}   {np.rad2deg(np.angle(Vph_mat[i,2])):.3f} |')
    print('=================================================================')  
    return
# %%
