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
    print('=================================================================')
    print('|                  Fault Analysis Results                       |')
    print('=================================================================')

    if fault_type == 0:
        print(f'| 3-Phase Balanced Fault at Bus {fault_bus}                           |')
    elif fault_type == 1:
        print(f'|Single Line-to-Ground fault at Bus {fault_bus}, phase a                  |')
    elif fault_type == 2:
        print(f'|Line-to-Line fault at Bus {fault_bus}                                |')
    elif fault_type == 3:
        print(f'|Double Line-to-Ground fault at Bus {fault_bus}                       |')
    
    print(f'|Prefault Voltage: Vf = {Vf:.4f}  (pu)                            |')
    print(f'|Fault impedance:  Zf = {Zf:.4f}  (pu)                            |')

    Iph_mag = np.abs(Iph)
    Iph_phase_rad = np.angle(Iph)
    Iph_phase_deg = np.rad2deg(Iph_phase_rad)

    print('|Phase Currents ------------------------------------------------|')
    print('|--------------                                                 |')
    print('|     ---- Phase a ---- | ---- Phase b ---- | ---- Phase c ---- |')
    print('|     ----------------- | ----------------- | ----------------- |')
    print('|      Mag(pu) Ang(deg) |  Mag(pu) Ang(deg) |  Mag(pu) Ang(deg) |')
    print(f'|     {Iph_mag[0]:6.3f} {Iph_phase_deg[0]:6.0f}       {Iph_mag[1]:6.3f} {Iph_phase_deg[1]:4.0f}         {Iph_mag[2]:6.3f} {Iph_phase_deg[2]:6.0f}     |')
    print('================================================================')
    print('|Phase Line-to-Ground Voltages ---------------------------------|')
    print('|-----------------------------                                  |')
    print('|   | ---- Phase a ---- | ---- Phase b ---- | ---- Phase c ---- |')
    print('|Bus|-------------------|-------------------|-------------------|')
    print('|   |  Mag(pu) Ang(deg) |  Mag(pu) Ang(deg) |  Mag(pu) Ang(deg) |')
    print('|---| -------- -------- |  ------- -------- |  ------- -------- |')
    
    Vph_mag = np.abs(Vph_mat)
    Vph_phase_rad = np.angle(Vph_mat)
    Vph_phase_deg = np.rad2deg(Vph_phase_rad)

    for i in range(Vph_mat.shape[0]):
        print(f'| {i+1} | {Vph_mag[i,0]:6.3f} {Vph_phase_deg[i,0]:6.2f}       {Vph_mag[i,1]:6.3f}   {Vph_phase_deg[i,1]:6.2f}    {Vph_mag[i,2]:6.3f}   {Vph_phase_deg[i,2]:6.2f}   |')
    
    print('=================================================================')  
    return