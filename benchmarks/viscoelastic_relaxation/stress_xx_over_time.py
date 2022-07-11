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
         "ve_relaxation_dt125yr_dh10km"
        ]
tail = r"/statistics"

# The labels the graphs will get in the plot
labels = [
          'dt = 125yr, dh = 10km'
         ]
# Set the colors available for plotting
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']


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
  plt.plot(time/1e3,stress_xx_min/1e6,label=labels[counter],color=colors[counter])

# Plot the analytical solution
# tau_xx(t) = tau_xx_t0 * exp(-mu*t/eta_viscous), 
# with tau_xx_t0 = 20 MPa, eta_viscous = 1e22 Pas, mu = 1e10 Pa.
yr_in_secs = 3600. * 24. * 365.2425
plt.plot(time/1e3,20*np.exp(-1e10*time*yr_in_secs/1e22),label='analytical',color='black',linestyle='dashed')

# Labelling of plot
plt.xlabel("Time [ky]")
plt.ylabel(r"Viscoelastic stress $\tau_{xx}$ [MPa]")
# Manually place legend in lower right corner. 
plt.legend(loc='upper right')
# Grid and tickes
plt.grid(axis='x',color='0.95')
plt.yticks([0,5,10,15,20])
plt.grid(axis='y',color='0.95')

# Ranges of the axes
plt.xlim(0,250) # kyr
plt.ylim(0,21) # MPa

plt.tight_layout()

# Save as pdf
plt.savefig('1_viscoelastic_relaxation.pdf')    
