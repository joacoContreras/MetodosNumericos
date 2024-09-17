import numpy as np
import matplotlib.pyplot as plt

# Sample data points (x, y)
# Replace these with your data
x = np.array([0.5, 0.8, 1.3, 2.0])
y = np.array([-0.716, -0.103, 3.419, 52.598])

# Calculate the mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Calculate the terms needed for the slope (a) and intercept (b)
numerator = np.sum((x - mean_x) * (y - mean_y))
denominator = np.sum((x - mean_x) ** 2)

# Calculate the slope (a)
a = numerator / denominator

# Calculate the intercept (b)
b = mean_y - a * mean_x

print(f"a = {a}, b = {b}")

# Generate the best-fit line
best_fit_line = a * x + b

# Plotting the data points and the best-fit line
plt.scatter(x, y, label='Data points')
plt.plot(x, best_fit_line, label='Best fit line', color='red')
plt.legend()
plt.show()
