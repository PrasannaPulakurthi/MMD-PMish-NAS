from sympy import symbols, Function, tanh, diff, exp, ln, sech

# Define symbols
x, beta = symbols('x beta')
f = Function('f')

# Define the PMish function
PMish = x * tanh(ln(1 + exp(beta * x)) / beta)

# First derivative
PMish_prime = diff(PMish, x).simplify()

# Second derivative
PMish_double_prime = diff(PMish_prime, x).simplify()

print(PMish)

print(PMish_prime)

print(PMish_double_prime)
