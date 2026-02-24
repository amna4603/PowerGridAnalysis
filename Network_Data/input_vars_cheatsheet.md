definition of the input parameters of the grid's components

--------------------------------------------------------
// MVA SYSTEM BASE DATA
- MVA_base (Float): The global apparent power base for the entire system, given in MVA. All per-unit conversions use this base value.

--------------------------------------------------------
// BUS DATA,(BUS_NR, LABEL, Voltage magnitude[pu], Voltage angle [degree], BUSCODE [1: PQ, 2: PV, 3: ref] V_rated [kV)], V_min [pu], V_max [pu])
- BUS_NR (Integer): The unique bus identifier.
- LABEL (String): A descriptive name for the bus.
- Voltage magnitude (Float): The initial voltage magnitude guess in per-unit (p.u.).
- Voltage angle (Float): The initial voltage angle guess in degrees.
- BUSCODE (Integer): Defines the type of bus for the power flow calculation: 1 = PQ (Load) bus, 2 = PV (Generator) bus, 3 = Reference/Slack bus.
- V_rated (Float): The nominal base voltage of the bus in kV.
- V_min (Float): The minimum acceptable operational voltage limit in p.u.
- V_max (Float): The maximum acceptable operational voltage limit in p.u.

--------------------------------------------------------
// LOAD DATA (BUS_NR, P_load MW, Q_load MVAR)
- BUS_NR (Integer): The bus to which the load is connected.
- P_load (Float): The active power consumed by the load in MW.
- Q_load (Float): The reactive power consumed by the load in MVAr.

--------------------------------------------------------
// GENERATOR DATA (BUS_NR, MVA_SIZE, P_GEN [MW], P_max [MW], Q_max [MVAr], Q_min [MVAr], X, X2, X0, Xn, GRND) // GRND: 1=grounded; 0:ungrounded
- BUS_NR (Integer): The unique identifier of the bus to which the generator is connected. This maps the generator to a specific node in the network model.
- MVA_SIZE (Float): The apparent power rating of the generator in MVA. This value is used as the base for calculating percentage loading and for normalizing specific limits
- P_GEN (Float): The active power generation setpoint in MW. In the LoadNetworkData script, this value is divided by the system MVA_base to convert it into per-unit (p.u.) for the Sbus injection vector.
- P_max (Float): The maximum active power capability of the generator in MW. This is used to check for generator overloads or limit violations after the power flow solution is obtained.
- Q_max (Float): The maximum reactive power capability in MVAr. This defines the upper limit for reactive power generation, used for checking operational constraints.
- Q_min (Float): The minimum reactive power capability in MVAr. This defines the lower limit (under-excitation limit), also used for checking constraints.
- X (Float): The positive sequence reactance (or synchronous reactance) of the generator in per unit (p.u.). While included in the data structure, it is primarily relevant for fault analysis rather than the steady-state Newton-Raphson power flow formulation used in Exercise 1.
- X2 (Float): The negative sequence reactance in p.u. Used for analyzing unbalanced faults (e.g., in future assignment modules).
- X0 (Float): The zero sequence reactance in p.u. Used for analyzing ground faults.
- Xn (Float): The neutral grounding reactance in p.u. This represents the impedance between the generator neutral and ground.
- GRND (Boolean/Integer): A flag indicating the grounding configuration of the generator.

--------------------------------------------------------
// LINE DATA (FROM_BUS, TO_BUS, ID, R, X, B, MVA_rating, X2, X0)
- FROM_BUS (Integer): The starting bus number of the transmission line.
- TO_BUS (Integer): The ending bus number of the line.
- ID (String): A unique branch identifier (useful for parallel lines).
- R (Float): The series resistance in p.u.
- X (Float): The series reactance in p.u.
- B (Float): The total shunt susceptance in p.u.
- MVA_rating (Float): The maximum thermal capacity limit of the line in MVA.
- X2 (Float): The negative sequence reactance (used in fault analysis).
- X0 (Float): The zero sequence reactance (used in ground fault analysis).

--------------------------------------------------------
// TRANSFORMER DATA (FROM_BUS, TO_BUS, ID, R, X, n, ANG1, MVA_rating, FROM_CON, TO_CON, X2, X0)
- FROM_BUS (Integer): The bus connected to the primary winding.
- TO_BUS (Integer): The bus connected to the secondary winding.
- ID (String): A unique branch identifier.
- R (Float): The equivalent resistance in p.u.
- X (Float): The equivalent reactance in p.u.
- n (Float): The off-nominal turns ratio or tap setting.
- ANG1 (Float): The phase shift angle in degrees (used for phase-shifting transformers).
- MVA_rating (Float): The power rating of the transformer in MVA.
- FROM_CON (Integer): The primary winding connection type (1=Y, 2=Y-grounded, 3=Delta).
- TO_CON (Integer): The secondary winding connection type (1=Y, 2=Y-grounded, 3=Delta).
- X2 (Float): Negative sequence reactance.
- X0 (Float): Zero sequence reactance.
