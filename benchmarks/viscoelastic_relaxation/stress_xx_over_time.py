# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 by Anne Glerum
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc("pdf", fonttype=42)
rc("lines", linewidth=5)

# Change path as needed
base = r"/Users/acglerum/Documents/Postdoc/SB_CRYSTALS/HLRN/HLRN/fix_stresses_elasticity/paper_11072022/"

# Change file name modifiers as needed depending on your file structure
names = [
         "ve_relaxation_dt125yr_dh10km",
         "ve_relaxation_dt125yr_dh10km_100NI",
         "VE_ve_relaxation_dt125yr_dh10km"
        ]
tail = r"/statistics"

# The labels the graphs will get in the plot
labels = [
          'dt = 125yr, dh = 10km',
          'dt = 125yr, dh = 10km, NI = 100',
          'dt = 125yr, dh = 10km, VE'
         ]
# Set the colors available for plotting
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']

# Set up a row of two plots, one with absolute stress values
# and one with the percentage difference to the analytical solution
fig = plt.figure(figsize=(10, 6))
ax = [fig.add_subplot(2, 1, i) for i in range(1, 3)]

yr_in_secs = 3600. * 24. * 365.2425
counter = 0 

# Create file path
for name in names: 
  path = base+name+tail

  # Read in the time and the minimum xx and yy components of the viscoelastic stress,
  # which are stored on the fields ve_stress_xx and ve_stress_yy.
  # The correct columns are selected with usecols.
  time,stress_xx_min = np.genfromtxt(path, comments='#', usecols=(1,15), unpack=True)

  # Plot the stress elements in MPa against time in ky in
  # categorical batlow colors.
  ax[0].plot(time/1e3,stress_xx_min/1e6,label=labels[counter],color=colors[counter])
  ax[1].plot(time/1e3,(stress_xx_min-(20e6*np.exp(-1e10*time*yr_in_secs/1e22)))/(20e6*np.exp(-1e10*time*yr_in_secs/1e22))*100.,label=labels[counter],color=colors[counter])
  
  counter += 1

# Plot the analytical solution
# tau_xx(t) = tau_xx_t0 * exp(-mu*t/eta_viscous), 
# with tau_xx_t0 = 20 MPa, eta_viscous = 1e22 Pas, mu = 1e10 Pa.
ax[0].plot(time/1e3,20*np.exp(-1e10*time*yr_in_secs/1e22),label='analytical',color='black',linestyle='dashed')

# Labelling of plot
ax[1].set_xlabel("Time [ky]")
ax[0].set_ylabel(r"Viscoelastic stress $\tau_{xx}$ [MPa]")
ax[1].set_ylabel(r"Error [%]")
# Manually place legend in lower right corner. 
ax[0].legend(loc='upper right')
ax[1].legend(loc='lower right')
# Grid and tickes
ax[0].grid(axis='x',color='0.95')
ax[0].set_yticks([0,5,10,15,20])
ax[0].grid(axis='y',color='0.95')
ax[1].grid(axis='x',color='0.95')
ax[1].grid(axis='y',color='0.95')
ax[1].set_yticks([0,0.5,1,1.5,2])

# Ranges of the axes
ax[0].set_xlim(0,250) # kyr
ax[0].set_ylim(0,21) # MPa
ax[1].set_xlim(0,250) # kyr
ax[1].set_ylim(0,2) # %

# Add labels a) and b)
ax[0].text(-13,20,"a)")
ax[1].text(-13,1.8,"b)")

plt.tight_layout()

# Save as pdf
plt.savefig('1_viscoelastic_relaxation.pdf')    
