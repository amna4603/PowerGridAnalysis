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
        Iseq[1] = Vf/Zbus1[fb,fb] + Zf
        print(f'Zbus1[fb,fb]={Zbus1[fb,fb]:.4f}, Zf={Zf:.4f}, Iseq[1]={Iseq[1]:.4f}')
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
 
    ZERO_THRESHOLD = 1e-6  # magnitudes below this are treated as zero
 
    def clean(val):
        mag = abs(val)
        ang = np.rad2deg(np.angle(val))
        if mag < ZERO_THRESHOLD:
            mag, ang = 0.0, 0.0
        return mag, ang
 
    Iph_clean = [clean(Iph[k]) for k in range(3)]
 
    fault_phases = ''
    if Iph_clean[0][0] != 0: fault_phases += 'a'
    if Iph_clean[1][0] != 0: fault_phases += 'b'
    if Iph_clean[2][0] != 0: fault_phases += 'c'
 
    # ── Layout constants ─────────────────────────────────────────────
    # Total line = 67 chars.  Structure: |<inner 65 chars>|
    # Phase section (no bus col):  | CA | CB | CC |
    #   1 + CA + 1 + CB + 1 + CC + 1 = 65  -> CA=CB=CC=20, each col 20 chars (incl leading |)
    # Voltage section (with bus col): | BUS | CA | CB | CC |
    #   1 + 3 + 1 + CA + 1 + CB + 1 + CC + 1 = 65 -> CA=CB=CC=19
 
    SEP  = '=' * 67          # full-width separator
    DASH = '-' * 67          # used nowhere but kept for reference
 
    def r(inner):
        """Print |<inner exactly 65 chars>|"""
        assert len(inner) == 65, f'Inner width={len(inner)}: "{inner}"'
        print(f'|{inner}|')
 
    def text_row(text):
        r(text[:65].ljust(65))
 
    # ── Data formatter (fixed width) ─────────────────────────────────
    def fmt(m, a, w):
        """Format magnitude+angle centred in w chars."""
        s = f'{m:6.3f}  {a:7.2f}'   # 15 chars
        return s.center(w)
 
    # ── Phase-current section helpers (col width = 20, no bus prefix)
    # inner = 20 + 1 + 20 + 1 + 20 + 1 + 1(leading space) ... recalc:
    # | SP CA | CB | CC |  where SP=1, CA=19, CB=19, CC=19 gives 1+19+1+19+1+19+1=61 -> too short
    # Use CA=CB=CC=21:  1+21+1+21+1+21 = 66 -> 1 too many
    # Use 1 leading space + 20+|+20+|+20+| = 1+20+1+20+1+20+1 = 64 -> 1 short
    # Best: leading | already in print, so inner=65:
    #   space(1) + col(20) + |(1) + col(20) + |(1) + col(20) + |(1) = 64 -> 1 short
    #   space(2) + col(20) + |(1) + col(20) + |(1) + col(20) + |(1) = 65 ✓
    CA = 20  # phase column content width (current section)
 
    def cur_sep():
        r(' ' + '-'*CA + '|' + '-'*CA + '|' + '-'*CA + '|' + '-')
        # 1 + 20 + 1 + 20 + 1 + 20 + 1 + 1 = 65 ✓... let's count: ' '=1, '-'*20=20, |=1, '-'*20=20, |=1, '-'*20=20, |=1, '-'=1 = 65 ✓
 
    def cur_phase_hdr():
        def ph(): return f'{"---- Phase a ----":^{CA}}'
        labels = ['Phase a','Phase b','Phase c']
        cols = [f'---- {l} ----'.center(CA) for l in labels]
        r(' ' + cols[0] + '|' + cols[1] + '|' + cols[2] + '|' + ' ')
 
    def cur_col_hdr():
        h = 'Mag(pu) Ang(deg)'.center(CA)
        r(' ' + h + '|' + h + '|' + h + '|' + ' ')
 
    def cur_data(ma, aa, mb, ab, mc, ac):
        r(' ' + fmt(ma,aa,CA) + '|' + fmt(mb,ab,CA) + '|' + fmt(mc,ac,CA) + '|' + ' ')
 
    # ── Voltage section helpers (bus col=4, then col width=19 each)
    # |BBBB|CA|CB|CC|  inner=65:  4+1+19+1+19+1+19+1 = 65 ✓
    BP = 4   # bus prefix width (e.g. ' 1  ' or 'Bus ' or '----')
    CV = 19  # voltage phase column content width
 
    def vol_sep(bus_prefix):
        r(bus_prefix[:BP] + '|' + '-'*CV + '|' + '-'*CV + '|' + '-'*CV + '|')
 
    def vol_phase_hdr(bus_prefix):
        labels = ['Phase a','Phase b','Phase c']
        cols = [f'---- {l} ----'.center(CV) for l in labels]
        r(bus_prefix[:BP] + '|' + cols[0] + '|' + cols[1] + '|' + cols[2] + '|')
 
    def vol_col_hdr(bus_prefix):
        h = 'Mag(pu) Ang(deg)'.center(CV)
        r(bus_prefix[:BP] + '|' + h + '|' + h + '|' + h + '|')
 
    def vol_data(bus_nr, ma, aa, mb, ab, mc, ac):
        bp = f' {bus_nr:2d} '
        r(bp + '|' + fmt(ma,aa,CV) + '|' + fmt(mb,ab,CV) + '|' + fmt(mc,ac,CV) + '|')
 
    # ── Print the table ──────────────────────────────────────────────
    print(SEP)
    text_row(f' {"Fault Analysis Results":^63}')
    print(SEP)
    text_row(f' {fault_types[fault_type]} at Bus {fault_bus}, phase {fault_phases}.')
    text_row(f' Prefault Voltage: Vf = {Vf:.3f}  (pu)')
    text_row(f' Fault Impedance:  Zf = {Zf:.3f}  (pu)')
    print(SEP)
    text_row(f' Phase Currents {"-" * 50}')
    cur_phase_hdr()
    cur_sep()
    cur_col_hdr()
    cur_sep()
    Ima,Iaa = Iph_clean[0]; Imb,Iab = Iph_clean[1]; Imc,Iac = Iph_clean[2]
    cur_data(Ima,Iaa,Imb,Iab,Imc,Iac)
    print(SEP)
    text_row(f' Phase Line-to-Ground Voltages {"-" * 35}')
    print(SEP)
    vol_phase_hdr('    ')
    vol_sep('Bus ')
    vol_col_hdr('    ')
    vol_sep('----')
    for i in range(Vph_mat.shape[0]):
        Vma,Vaa = clean(Vph_mat[i,0])
        Vmb,Vab = clean(Vph_mat[i,1])
        Vmc,Vac = clean(Vph_mat[i,2])
        vol_data(i+1, Vma,Vaa, Vmb,Vab, Vmc,Vac)
    print(SEP)
    return
