import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Define symbols
x, beta = sp.symbols('x beta')

# Define the PMish function
PMish = x * sp.tanh(sp.ln(1 + sp.exp(beta * x)) / beta)

# First derivative
PMish_prime = sp.diff(PMish, x).simplify()

# Second derivative
PMish_double_prime = sp.diff(PMish_prime, x).simplify()

# Lambdify the expressions for numerical evaluation
PMish_prime_func = sp.lambdify((x, beta), PMish_prime, 'numpy')
PMish_double_prime_func = sp.lambdify((x, beta), PMish_double_prime, 'numpy')

# Define range and values for x
x_vals = np.linspace(-6, 6, 400)

# Define different beta values
beta_values = [0.5, 1, 2, 5]
colors = ['purple', 'green', 'blue', 'red']

# Save the plots into separate files
first_derivative_path = 'first_derivative_pmish.png'
second_derivative_path = 'second_derivative_pmish.png'
# First derivative plot
plt.figure(figsize=(6, 6))
for beta, color in zip(beta_values, colors):
    plt.plot(x_vals, PMish_prime_func(x_vals, beta), color=color, label=f'β = {beta}')
plt.title('First Derivative of Parametric Mish')
plt.xlabel('x')
plt.ylabel("f '(x)")
plt.legend(loc ="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(first_derivative_path)

# Second derivative plot
plt.figure(figsize=(6, 6))
for beta, color in zip(beta_values, colors):
    plt.plot(x_vals, PMish_double_prime_func(x_vals, beta), color=color, label=f'β = {beta}')
plt.title('Second Derivative of Parametric Mish')
plt.xlabel('x')
plt.ylabel("f ''(x)")
plt.legend(loc ="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(second_derivative_path)

