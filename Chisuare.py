import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
sns.set()

# Read the data
df = pd.read_csv('data/intent99.7.csv')

# Plot the observed data
ax = df.plot(x='Position', y='Observed', kind='scatter', label='Exp')

# Parameters
w0 = 30e-4
wavelength = 1040e-7
z0 = np.pi * w0**2 / wavelength

# Define the function for fitting
def func(z_values, phi01):
    x = z_values / z0
    return 1 + (4 * x * phi01) / ((x**2 + 1) * (x**2 + 9))

# Fit the function to the observed data
popt, pcov = curve_fit(func, df['Position'], df['Observed'])

# Calculate chi-square values
df['Chi-Square'] = ((df['Observed'] - func(df['Position'], *popt)) / 1)**2

# Plot the chi-square values
ax.plot(df['Position'], df['Chi-Square'], 'r.', label='Chi-Square')

# Set plot labels and legend
ax.set_xlabel('Position z (cm)')
ax.set_ylabel('Normalized Transmittance (a.u.)')
ax.legend()
ax.set_title(f'$\\phi_{{n}}$ = {popt[0]:.1e} $\\pm$ {np.sqrt(pcov[0, 0]):.1e}')

# Adjust plot limits
ax.set_ylim(0, max(df['Chi-Square']) + 0.1)

# Show the plot
plt.show()

# Print chi-square value
total_chi_square = df['Chi-Square'].sum()
print(f'Total Chi-Square: {total_chi_square:.4f}')
