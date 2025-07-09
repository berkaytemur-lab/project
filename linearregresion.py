import matplotlib.pyplot as plt
from scipy import stats

# Your data
xdat = [1, 2, 3, 4, 6, 8, 0, 2, 3, 6]
ydat = [3, 4, 3, 5, 6, 2, 4, 5, 1, 7]

# Perform linear regression
regression = stats.linregress(xdat, ydat)

# Generate predicted y values for the line
slope = regression.slope
intercept = regression.intercept
line = [slope * x + intercept for x in xdat]

# Plotting
plt.scatter(xdat, ydat, color='blue', label='Data Points')     # actual data
plt.plot(xdat, line, color='red', label='Regression Line')     # regression line
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
