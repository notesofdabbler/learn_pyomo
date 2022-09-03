
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

def rates(t, y):
    dy0 = -0.04 * y[0] + 10**4 * y[1] * y[2]
    dy1 = 0.04 * y[0] - 10**4 * y[1] * y[2] - 3 * 10**7 * y[1]**2
    dy2 = 3 * 10**7 * y[1]**2
    return [dy0, dy1, dy2]

sol = solve_ivp(rates, [0.0, 4.0e10],   [1.0, 0.0, 0.0], method = 'Radau')

fig, ax = plt.subplots()
ax.plot(np.log10(sol.t), sol.y[0, :])

fig, ax = plt.subplots()
ax.plot(np.log10(sol.t), sol.y[1, :])

fig, ax = plt.subplots()
ax.plot(np.log10(sol.t), sol.y[2, :])

fig, ax = plt.subplots()
ax.plot(sol.t, sol.y[0, :])


