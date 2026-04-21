import matplotlib.pyplot as plt
import numpy as np
import FaultAnalysis_46705 as fa    # import Fault Analysis functions
import LoadNetworkData4FA as lnd4fa # load the network data to global variables

filename = "./TestSystem4FA.txt"
lnd4fa.LoadNetworkData4FA(filename) # makes Zbus0 available as lndfa.Zbus0 etc.

# Carry out the fault analysis ...
def generate_and_save_voltage_impact_plots(lnd4fa):
    """
    Generates and saves 5 PNG figures (one for each bus).
    Each figure shows I_f as a function of V_f for all 4 fault types.
    """
    # Define the voltage range as requested (0.95 to 1.05 p.u.)
    v_range = np.linspace(0.95, 1.05, 10)
    
    # Fault types dictionary for labels
    fault_types = {
        0: "3-Phase Balanced", 
        1: "Single Line-to-Ground", 
        2: "Line-to-Line", 
        3: "Double Line-to-Ground"
    }
    
    # Fixed parameters for this analysis
    fault_impedance = 0  # Bolted fault assumption [cite: 2368, 3058]
    
    # Loop through each bus (1 to 5)
    for bus_num in range(1, 6):
        plt.figure(figsize=(10, 6))
        
        # Iterate through each fault type to plot 4 curves per figure
        for f_type, f_name in fault_types.items():
            max_currents = []
            
            for v_prefault in v_range:
                # Calculate fault currents using the systematic Z-Bus method [cite: 2765, 3431]
                Iph, _ = fa.FaultAnalysis(
                    lnd4fa.Zbus0, lnd4fa.Zbus1, lnd4fa.Zbus2, 
                    lnd4fa.bus_to_ind, bus_num, f_type, 
                    fault_impedance, v_prefault
                )
                
                # Take max among the 3 phases [cite: 3272, 3343]
                max_currents.append(np.max(np.abs(Iph)))
            
            plt.plot(v_range, max_currents, label=f_name, linewidth=2)
        
        # Formatting the figure
        plt.title(f"Fault Current Magnitude vs. Pre-fault Voltage (Bus {bus_num})")
        plt.xlabel("Pre-fault Voltage $V_f$ [p.u.]")
        plt.ylabel("Max Phase Current $I_{f,max}$ [p.u.]")
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        
        # Save the figure as PNG without displaying it
        filename = f"Bus{bus_num}_Voltage_Impact.png"
        plt.savefig(filename, dpi=300)
        plt.close()  # Close to free up memory
        
        print(f"Successfully saved: {filename}")

# Note: Ensure lnd4fa and fa are initialized before calling
generate_and_save_voltage_impact_plots(lnd4fa)
# %%
