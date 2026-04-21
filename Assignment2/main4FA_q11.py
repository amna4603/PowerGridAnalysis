import matplotlib.pyplot as plt
import numpy as np
import FaultAnalysis_46705 as fa    # import Fault Analysis functions
import LoadNetworkData4FA as lnd4fa # load the network data to global variables

filename = "./TestSystem4FA.txt"
lnd4fa.LoadNetworkData4FA(filename) # makes Zbus0 available as lndfa.Zbus0 etc.

# Carry out the fault analysis ... 
def generate_fault_impedance_plot(fault_type, z_range, label, lnd4fa, fa):
    """
    Generates data and plots fault current vs impedance for a specific fault type at Bus 4.
    fault_type: 0=3-phase, 1=SLG, 2=LL, 3=DLG
    z_range: array of impedance values (resistive or reactive)
    """
    fault_bus = 4
    prefault_voltage = 1.0  # Standard prefault voltage used in examples [cite: 2729, 2889]
    currents_pu = []
    voltages_pu = []

    for z_val in z_range:
        # Determine if we treat input as R or X based on typical literature
        # 3-phase faults often focus on Xf; unbalanced on Rf
        fault_impedance = complex(0, z_val) if fault_type == 0 else complex(z_val, 0)
        
        # Calculate currents using the existing FaultAnalysis function
        Iph, Vph_mat = fa.FaultAnalysis(lnd4fa.Zbus0, lnd4fa.Zbus1, lnd4fa.Zbus2, 
                                        lnd4fa.bus_to_ind, fault_bus, fault_type, 
                                        fault_impedance, prefault_voltage)
        
        # Take the maximum phase current magnitude for the plot
        currents_pu.append(np.max(np.abs(Iph)))
        # Take the minimum phase voltage magnitude at the fault bus for the plot
        voltages_pu.append(np.min(np.abs(Vph_mat[lnd4fa.bus_to_ind[fault_bus]])))

    return currents_pu, voltages_pu

# Setup for the plots
fault_names = {0: "3-Phase Balanced (Reactive)", 
               1: "Single Line-to-Ground (Resistive)", 
               2: "Line-to-Line (Resistive)", 
               3: "Double Line-to-Ground (Resistive)"}

# Define typical ranges based on literature talked about earlier:
# 3-phase: 0 to 0.1 pu; Unbalanced: 0 to 0.5 pu
z_ranges = {
    0: np.linspace(0, 0.1, 100),
    1: np.linspace(0.05, 0.5, 100),
    2: np.linspace(0., 0.2, 100),
    3: np.linspace(0.05, 0.5, 100)
}

plt.figure(figsize=(12, 12))

# Plot for fault currents
plt.subplot(2, 1, 1)
for f_type in range(4):
    currents, _ = generate_fault_impedance_plot(f_type, z_ranges[f_type], fault_names[f_type], lnd4fa, fa)
    plt.plot(z_ranges[f_type], currents, label=fault_names[f_type])

# plt.title("Impact of Fault Impedance on Subtransient Fault Current at Bus 4") # title in the latex
plt.xlabel("Fault Impedance Magnitude (p.u.)", fontsize=14)
plt.ylabel("Maximum Fault Current Magnitude (p.u.)", fontsize=14)
plt.legend()
plt.grid(True)

# Plot for fault voltages
plt.subplot(2, 1, 2)
for f_type in range(4):
    _, voltages = generate_fault_impedance_plot(f_type, z_ranges[f_type], fault_names[f_type], lnd4fa, fa)
    plt.plot(z_ranges[f_type], voltages, label=fault_names[f_type])

# plt.title("Impact of Fault Impedance on Bus 4 Voltage") # title in the latex
plt.xlabel("Fault Impedance Magnitude (p.u.)", fontsize=14)
plt.ylabel("Minimum Voltage Magnitude at Bus 4 (p.u.)", fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# %%
