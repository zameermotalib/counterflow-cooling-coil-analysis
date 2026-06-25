# counterflow-cooling-coil-analysis
Iterative numerical model of a counterflow cooling and dehumidifying coil, coupling refrigerant and moist-air heat and mass transfer using CoolProp
# Counterflow Cooling Coil Analysis

A discretised, iterative numerical model of a counterflow heat exchanger 
(refrigerant side and moist-air side), built for a thermofluids assignment. 
Solves coupled heat and mass transfer along the length of the coil, including 
condensation on the air side.

## What it does
- Discretises the heat exchanger into segments and sets up linear balance 
  equations for refrigerant enthalpy, air enthalpy, and air humidity
- Solves the coupled system iteratively with under-relaxation for stability
- Accounts for condensate formation when the air is cooled below its dew point
- Outputs total heat transfer, UA value, condensate rate, and inlet/outlet 
  states
- Plots the temperature distribution along the coil and a full psychrometric 
  chart showing the air process path

## Tools
Python, NumPy, Matplotlib, CoolProp (for refrigerant and moist-air properties)

## What I learned
This was my first real experience with an iterative numerical solver that 
had to balance multiple coupled physical processes (sensible heat, latent 
heat, mass transfer) at once. Getting the under-relaxation factor right to 
keep the solution stable, without making it painfully slow to converge, 
taught me a lot about the practical side of numerical methods beyond the 
theory.
