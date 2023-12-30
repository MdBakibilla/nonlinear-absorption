import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2

sns.set()

# Load data
df = pd.read_csv('/home/bakibilla/Desktop/project/intensity339.csv')

# Create a white background
sns.set_style("white")

# Create the plot
ax = df.plot(x='Position', y='Observed', kind='scatter', label='Experimental I$_0$ = 339 GW/cm$^2$')

# Define the parameters
n = 3
I0 = 129 * 10**9
Leff = 1.00
w0 = 30e-4
wavelength = 1040e-7
z0 = np.pi * w0**2 / wavelength

def func(z_values, alpha):
    x_values = z_values / z0
    return 1 - alpha * (I0**(n - 1) * Leff) / (n**(3 / 2) * (1 + x_values**2)**(n - 1))

# Values for the curve
z_values = np.linspace(-3, 3, 1500)

# Fit the curve
popt, pcov = curve_fit(func, df['Position'].T, df['Observed'])
ax.plot(z_values, func(z_values, popt[0]), 'b-', label='Fitted Curve n = '+str(n), linewidth=2)

# Calculate standard deviation
stdev = np.sqrt(np.diag(pcov))

# Calculate the difference
df['Difference'] = (df['Observed'] - func(df['Position'], *popt))**2 + 1

# Plot the difference
df.plot(x='Position', y='Difference', style='^', label='Difference of Observed & Fit', color='r', ax=ax)

# Set font sizes
plt.xlabel('Position z, in cm', fontsize=14)
plt.ylabel('Normalized Transmittance in a.u.', fontsize=14)
plt.legend(fontsize=14)
plt.title('$\\alpha_{}$ = {:.1e} $\pm$ {:.1e}'.format(n, popt[0], stdev[0]), fontsize=16)
plt.ylim(0.8, 1.04)

# Calculate alpha
tn0 = df.Observed.min()
alpha = (1 - tn0) * n**(3/2) / (I0**(n-1) * Leff)

# Plot theoretical fit
ax.plot(z_values, func(z_values, alpha), 'g-', label='Theoretical Fit n = '+str(n), linewidth=2)

# Print values
print(tn0, alpha)

# Calculate difference using alpha
df['Difference'] = (df['Observed'] - func(df['Position'], alpha))**2 + 1

# Plot the difference using alpha
df.plot(x='Position', y='Difference', style='.', label='Difference Using $\\alpha_{}$ ', color='g', ax=ax)

# Calculate residuals
residuals = df['Observed'] - func(df['Position'], *popt)

# Calculate chi-square value
chi_square = np.sum(residuals ** 2 / func(df['Position'], *popt))
print("Chi-square:", chi_square)

# Adjust the right margin and maximize space for plotting
plt.subplots_adjust(right=0.95)

# Set overall font size
plt.rcParams.update({'font.size': 16})

plt.title('$\\alpha_{}$ = {:.1e} $\pm$ {:.1e}, $\\chi^2$ = {:.5f}'.format(n, popt[0], stdev[0], chi_square))
plt.legend(loc='lower left')
ax.tick_params(axis='both', which='major', labelsize=14)

# Degrees of freedom df  = n-k where n = no. of Observence k = the number of parameters estimated by the model during the curve fitting process.
df_degrees_of_freedom = len(df['Observed']) - len(popt)


# Significance level (alpha)
sig_alpha= 0.05

# Critical chi-square value
critical_chi_square = chi2.ppf(1 - sig_alpha, df_degrees_of_freedom)

# Calculate critical p-value
critical_p_value = 1 - chi2.cdf(chi_square, df_degrees_of_freedom)

print("Critical chi-square value:", critical_chi_square)
print("Critical p-value:", critical_p_value)

# Show the plot
plt.show()


