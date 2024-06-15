
import numpy as np
import matplotlib.pyplot as plt

# Define the parametric mish activation function
def parametric_mish(x, beta):
    return x * np.tanh((1/beta) * np.log1p(np.exp(beta * x)))

# Define the range of x values
x_values = np.linspace(-10, 10, 400)

# Define the beta values and corresponding colors
beta_values = [0.5, 1, 2, 5]
colors = ['purple', 'green', 'blue', 'red']  # Swapped order of colors

# Plot the functions
plt.figure(figsize=(10, 6))
for beta, color in zip(beta_values, colors):
    y_values = parametric_mish(x_values, beta)
    plt.plot(x_values, y_values, label=f'\beta = {beta}', color=color)

plt.title('Parametric Mish Activation Function')
plt.xlabel('x')
plt.ylabel('PMish(x)')
plt.legend()
plt.grid(True)
plt.show()
