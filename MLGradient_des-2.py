import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return (x + 3) ** 2

def derivative(x):
    return 2 * (x + 3)


def gradient_descent(starting_point, learning_rate, num_iterations):
    x = starting_point
    history = []  
    for _ in range(num_iterations):
        x -= learning_rate * derivative(x)
        history.append(x)
    return x, history


starting_point = 2
learning_rate = 0.1
num_iterations = 100


minima, history = gradient_descent(starting_point, learning_rate, num_iterations)


print(f"Local minima found at x = {minima:.4f}, y = {function(minima):.4f}")


x_values = np.linspace(-6, 2, 100)
y_values = function(x_values)

plt.plot(x_values, y_values, label='y = (x + 3)²')
plt.scatter(history, function(np.array(history)), color='red', label='Descent Path')
plt.title('Gradient Descent to Find Local Minima')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
