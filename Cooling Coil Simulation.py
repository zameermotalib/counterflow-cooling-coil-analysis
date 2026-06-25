import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from CoolProp.CoolProp import HAPropsSI
import numpy as np
import matplotlib.pyplot as plt
import copy

K = 273.15
fluid = 'Water'
mdot_r = 6.151    # mass flow rate [kg/s]
P_r = 300e3       # pressure [Pa]
T_r_in = 3.97 + K  # inlet temperature [K]
h_r_in = PropsSI('H', 'T', T_r_in, 'P', P_r, fluid)  # inlet enthalpy
p = 95e3          # static pressure [Pa]
mdot_a = -7.353   # mass flow rate [kg/s] (negative = counterflow)
T_a_in = 23.91 + K  # inlet dry bulb temperature [K]
w_in = 0.00986    # inlet humidity ratio [kg/kg_a]
h_a_in = HAPropsSI('H', 'T', T_a_in, 'W', w_in, 'P', p)  # inlet enthalpy
UA_total = 9.456e3  # total UA value [W/K]

# Initial guess - refrigerant outlet
T_r_out = T_r_in + 5.5  # K
h_r_out = PropsSI('H', 'T', T_r_out, 'P', P_r, fluid)

# Initial guess - air outlet
T_a_out = 11.0 + K   # K
w_out = 0.00741      # kg/kg_a
h_a_out = HAPropsSI('H', 'T', T_a_out, 'W', w_out, 'P', p)

# Discretisation - number of increments along the HX
n_tot = 50
# Refrigerant side - linear distribution between inlet and outlet guess
T_r = np.linspace(T_r_in, T_r_out, n_tot + 1)
h_r = np.linspace(h_r_in, h_r_out, n_tot + 1)
# Air side - note reversed direction (counterflow arrangement)
T = np.linspace(T_a_out, T_a_in, n_tot + 1)
h = np.linspace(h_a_out, h_a_in, n_tot + 1)
w = np.linspace(w_out, w_in, n_tot + 1)
T_r_ave = np.zeros(n_tot + 1)
T_ave = np.zeros(n_tot + 1)
w_ave = np.zeros(n_tot + 1)
T_w_ave = np.zeros(n_tot + 1)  # water film temperature (= refrigerant, simplifying assumption)
w_w_ave = np.zeros(n_tot + 1)  # saturation humidity at water film surface

# Populate initial cell-averaged values from the linear guesses
for i in range(1, n_tot + 1):
    T_ave[i] = 0.5 * (T[i - 1] + T[i])
    w_ave[i] = 0.5 * (w[i - 1] + w[i])
    T_r_ave[i] = 0.5 * (T_r[i - 1] + T_r[i])
    T_w_ave[i] = T_r_ave[i]
    w_w_ave[i] = HAPropsSI('W', 'T', T_w_ave[i], 'RH', 1.0, 'P', p)

Q_dot_s = np.zeros(n_tot + 1)   # sensible heat transfer rate
Q_dot = np.zeros(n_tot + 1)     # total heat transfer rate (sensible + latent)
Q_dot_r = np.zeros(n_tot + 1)   # refrigerant-side heat transfer rate
m_dot_w = np.zeros(n_tot + 1)   # condensate mass transfer rate
rho_ave = np.zeros(n_tot + 1)
cp_ave = np.zeros(n_tot + 1)
h_cA = np.zeros(n_tot + 1)
h_mA = np.zeros(n_tot + 1)
UA = np.zeros(n_tot + 1)
h_gw = np.zeros(n_tot + 1)   # saturated vapour enthalpy at film surface
h_fw = np.zeros(n_tot + 1)   # saturated liquid enthalpy at film surface
h_fgw = np.zeros(n_tot + 1)  # latent heat of vaporisation at film surface

dh = np.zeros(n_tot + 1)
dw = np.zeros(n_tot + 1)
dh_r = np.zeros(n_tot + 1)

err = np.zeros(n_tot + 1)

# Connectivity matrices for the balance equations
X_h = np.zeros((n_tot + 1, n_tot + 1))
Y_h = np.zeros(n_tot + 1)
X_w = np.zeros((n_tot + 1, n_tot + 1))
Y_w = np.zeros(n_tot + 1)
X_h_r = np.zeros((n_tot + 1, n_tot + 1))
Y_h_r = np.zeros(n_tot + 1)

for i in range(0, n_tot + 1):
    for j in range(0, n_tot + 1):
        if j == i:
            X_h[i, j] = -mdot_a
            X_w[i, j] = -mdot_a
            X_h_r[i, j] = mdot_r
        if j == i + 1:
            X_h[i, j] = mdot_a
            X_w[i, j] = mdot_a
        if j == i - 1:
            X_h_r[i, j] = -mdot_r

# Iterative solution loop
relax = 0.5   # under-relaxation factor
n_iter = 0    # iteration counter
residual = 1e10  # initialise residual to large value

while (residual > 1e-4 and n_iter < 50):
    n_iter += 1
    h_r_old = h_r
    h_old = h
    w_old = w
    for i in range(1, n_tot + 1):
        h_cA[i] = UA_total / n_tot  # Simplifying assumption
        UA[i] = h_cA[i]  # implifying assumption
        rho_ave[i] = 1 / HAPropsSI('V', 'T', T_ave[i], 'W', w_ave[i], 'P', p)
        cp_ave[i] = HAPropsSI('cp', 'T', T_ave[i], 'W', w_ave[i], 'P', p)
        h_mA[i] = h_cA[i] / (rho_ave[i] * cp_ave[i])
        Q_dot_s[i] = h_cA[i] * (T_w_ave[i] - T_ave[i])

        # Mass transfer from water film to moist air:
        if w_w_ave[i] - w_ave[i] > 0:
            m_dot_w[i] = 0.0
        else:
            m_dot_w[i] = rho_ave[i] * h_mA[i] * (w_w_ave[i] - w_ave[i])

        h_gw[i] = PropsSI('H', 'T', T_w_ave[i], 'Q', 1.0, 'Water')
        h_fw[i] = PropsSI('H', 'T', T_w_ave[i], 'Q', 0.0, 'Water')
        h_fgw[i] = h_gw[i] - h_fw[i]  # latent heat
        Q_dot[i] = Q_dot_s[i] + m_dot_w[i] * h_fgw[i]
        Q_dot_r[i] = -Q_dot[i]

        # Build source term vectors for the balance equations
        for i in range(0, n_tot):
            Y_h[i] = Q_dot_s[i + 1] + m_dot_w[i + 1] * h_gw[i + 1]
            Y_w[i] = m_dot_w[i + 1]

        # Air inlet boundary condition (known inlet state)
        Y_h[n_tot] = -mdot_a * h_a_in
        Y_w[n_tot] = -mdot_a * w_in
        Y_h_r[0] = mdot_r * h_r_in
        for i in range(1, n_tot + 1):
            Y_h_r[i] = Q_dot_r[i]

    # Solve the three linear systems simultaneously
    dh = np.matmul(np.linalg.inv(X_h), Y_h) - h_old
    dw = np.matmul(np.linalg.inv(X_w), Y_w) - w_old
    dh_r = np.matmul(np.linalg.inv(X_h_r), Y_h_r) - h_r_old
    # Apply under-relaxation to stabilise convergence
    h = h_old + relax * dh
    w = w_old + relax * dw
    h_r = h_r_old + relax * dh_r
    # Recover temperatures from updated enthalpies
    T = HAPropsSI('T', 'H', h, 'W', w, 'P', p)
    T_r = PropsSI('T', 'H', h_r, 'P', P_r, fluid)

    # Recompute cell-averaged quantities for next iteration
    for i in range(1, n_tot + 1):
        T_ave[i] = 0.5 * (T[i - 1] + T[i])
        w_ave[i] = 0.5 * (w[i - 1] + w[i])
        T_r_ave[i] = 0.5 * (T_r[i - 1] + T_r[i])
        T_w_ave[i] = T_r_ave[i]  # Simplifying assumption
        w_w_ave[i] = HAPropsSI('W', 'T', T_w_ave[i], 'RH', 1.0, 'P', p)

    for i in range(0, n_tot + 1):
        err[i] = max(np.abs((h[i] - h_old[i]) / h[i]), np.abs((w[i] - w_old[i]) / w[i]),
                        np.abs((h_r[i] - h_r_old[i]) / h_r[i]))
    residual = np.amax(err)
    print('it1 = {:3}'.format(n_iter), '    err 1 = {:0.3e}'.format(residual))

print('Solution completed')

# Post-processing - extract overall results
UA_total = np.sum(h_cA)
Q_dot_tot = np.sum(Q_dot)
m_dot_w_tot = np.sum(m_dot_w)

# Final outlet states
h_r_out = h_r[n_tot]
T_r_out = T_r[n_tot]
h_a_out = h[0]
w_out = w[0]
T_a_out = T[0]

# Cumulative UA along the heat exchanger (for plotting x-axis)
UA_x = np.zeros(n_tot + 1)
for i in range(1, n_tot + 1):
    UA_x[i] = UA_x[i - 1] + UA[i]

print('UA_tot = {:0.3f} [kW/K]'.format(UA_total / 1e3))
print('Q_dot_tot = {:0.3f} [kW]'.format(Q_dot_tot / 1e3))
print('m_dot_w_tot = {:0.4f} [kg/s]'.format(m_dot_w_tot))
print('T_i = {:0.3f} [\u00B0C]'.format(T_a_in - K))
print('T_e = {:0.3f} [\u00B0C]'.format(T_a_out - K))
print('w_i = {:0.5f} [kg/kg_a]'.format(w_in))
print('w_e = {:0.5f} [kg/kg_a]'.format(w_out))
print('T_ri = {:0.3f} [\u00B0C]'.format(T_r_in - K))
print('T_re = {:0.3f} [\u00B0C]'.format(T_r_out - K))

# Plot 1
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title('Temperature Distribution Along Heat Exchanger', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Position [kW/K]', fontsize=12)
ax.set_ylabel('Temperature [\u00B0C]', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
# Plot lines with explicit labels for the legend
ax.plot(UA_x / 1e3, T_r - K, color='steelblue', linestyle='-', lw=1.5, marker='o', markersize=3, label='Water')
ax.plot(UA_x / 1e3, T - K, color='tomato', linestyle='-', lw=1.5, marker='s', markersize=3, label='Moist Air')
# Enable the legend box
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
fig.tight_layout()
plt.show()

# Plot 2
def PsychroChart(Pressure, Drybulb, Humidity, T_min=0 + 273.15, T_max=40 + 273.15, w_max=0.03, dT=2, dw=0.002):
    # Function to plot a set of process variables on the Ts, ph, pv and hs diagrams showing the saturation region
    # The given set of process points are automatically connected with smooth lines (not just straight lines)
    # Pressure - the static pressure of the mixture in [Pa]
    # Drybulb - a Numpy array of drybulb temperature in [K] (index zero is also considered as a point)
    # Humidity - a Numpy array of absolute humidity values in [kg/kg_a] (index zero is also considered as a point)
    # T_min - minimum drybulb temperature limit on chart
    # T_max - maximum drybulb temperature on chart
    # w_max - maximum humidity on chart (minimum is always zero)
    # dT - drybulb temperature increment size on chart
    # dw - humidity increment size on chart
    # CnotK - 'TRUE' to plot temperatures in Celsius, otherwise 'FALSE' for Kelvin

    p = copy.deepcopy(Pressure)
    T_plot = copy.deepcopy(Drybulb)
    w_plot = copy.deepcopy(Humidity)
    w_max_abs = copy.deepcopy(w_max)

    if (len(T_plot) != len(w_plot)):
        message = 'Length of humodity array not the same as temperature array in PsychroChart'
        raise Exception(message)
    kel = 273.15
    n = 20
    T_db = np.linspace(T_min, T_max, num=n)
    fig = plt.figure(figsize=(10, 7))
    fig.tight_layout()
    plt.rc('font', size=11)  # controls default text sizes

    ax = fig.add_subplot(111)
    ax.set_xlabel('Dry Bulb Temperature [\u00B0C]', fontsize=12)
    ax.set_ylabel('Absolute Humidity [kg/kg$_{a}$]', fontsize=12)
    ax.set_title('Psychrometric Chart  |  p = {} kPa'.format(round(p / 1e3, 1)), fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(T_min - kel, T_max - kel)
    ax.set_ylim(0, w_max_abs)
    ax.set_yticks(np.arange(0, w_max_abs + dw, dw))
    ax.set_xticks(np.arange(T_min - kel, T_max - kel + dT, dT))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4)

    # Lines of constant relative humidity
    for RH in np.arange(0.1, 1, 0.1):
        w = CP.HAPropsSI("W", "R", RH, "P", p, "T", T_db)
        plt.plot(T_db - kel, w, color='grey', lw=0.5)

    # Saturation curve
    w = CP.HAPropsSI("W", "R", 1, "P", p, "T", T_db)
    plt.plot(T_db - kel, w, color='k', lw=0.7)

    # Lines of constant wetbulb
    for T_wb in np.arange(T_min, T_max, dT):
        T_min_wb = T_wb
        T_max_wb = min(T_max, CP.HAPropsSI("Tdb", "W", 0.0005, "Twb", T_wb, "P", p))
        T = np.linspace(T_min_wb, T_max_wb, num=2)
        w = CP.HAPropsSI("W", "Tdb", T, "Twb", T_wb, "P", p)
        plt.plot(T - kel, w, color='grey', lw=0.5)

    # Lines of constant drybulb
    for T_db in np.arange(T_min, T_max, dT):
        w_min = 0.0005
        w_max = CP.HAPropsSI("W", "Tdb", T_db, "R", 1, "P", p)
        T = np.linspace(T_db, T_db, num=2)
        w = np.linspace(w_min, w_max, num=2)
        plt.plot(T - kel, w, color='grey', linestyle='dashed', lw=0.5)

    # Right hand side drybulb line
    w_min = 0.0005
    w_max = CP.HAPropsSI("W", "Tdb", T_max, "R", 1, "P", p)
    T = np.linspace(T_max, T_max, num=2)
    w = np.linspace(w_min, w_max, num=2)
    plt.plot(T - kel, w, color='k', lw=0.5)

    # Lines of constant humidity
    w_min = dw
    w_max = np.floor(CP.HAPropsSI("W", "Tdb", T_max, "R", 1, "P", p) / 0.01) * 0.01
    for ww in np.arange(w_min, w_max + dw, dw):
        p_s = max((ww * p) / (0.622 + ww), 611.655)
        T_min_w = max(CP.PropsSI('T', 'P', p_s, 'Q', 0, 'Water'), T_min)
        T_max_w = T_max
        T = np.linspace(T_min_w, T_max_w, num=2)
        w = np.linspace(ww, ww, num=2)
        plt.plot(T - kel, w, color='grey', linestyle='dashed', lw=0.5)

    # Bottom humidity line
    T = np.linspace(T_min, T_max, num=2)
    w = np.linspace(0, 0, num=2)
    plt.plot(T - kel, w, color='k', lw=0.5)

    # Overlay the air-side process path
    plt.plot(T_plot - kel, w_plot, color='steelblue', linestyle='-', linewidth=1.5, marker='o', markersize=4, label='Air Process')
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.show()
    return
PsychroChart(p, T, w, T_min=8 + 273.15, T_max=28 + 273.15, w_max=0.012, dT=2, dw=0.002)