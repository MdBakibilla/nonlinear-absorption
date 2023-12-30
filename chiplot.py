import matplotlib.pyplot as plt

# First set of coordinates
coordinates1 = [
    (129, 0.00087),
    (249, 0.00453),
    (309, 0.01135),
    (339, 0.01326),
    (369, 0.01670)
]

# Second set of coordinates
coordinates2 = [
    (129, 0.00031),
    (249, 0.00292),
    (309, 0.00170),
    (339, 0.00184),
    (369, 0.00245)
    
]

# Extract x and y values for both sets of coordinates
x_values1, y_values1 = zip(*coordinates1)
x_values2, y_values2 = zip(*coordinates2)

# Plot the coordinates
plt.plot(x_values1, y_values1, marker='o',  color='b', label='Two Photon Absorption')
plt.plot(x_values2, y_values2, marker='o',  color='r', label='Three Photon Absorption')

# Add labels and title
plt.xlabel('Input Intensity')
plt.ylabel('$\chi^2$')
plt.title('Differences Between 2PA vs 3PA')

# Show the legend
plt.legend()

# Show the plot
plt.show()
