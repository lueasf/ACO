import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

T = 50
def trajectory(t):
    return np.array([np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T)])

def objective(x):
    return -x[0] - x[1]

def constraint(x, center):
    return 0.5 - np.linalg.norm(x - center) # circle of radius 0.5

x_opt = np.zeros((T, 2))
x0 = trajectory(0) + np.array([0.1, 0.0])

for t in range(T):
    center = trajectory(t)
    cons = {'type': 'ineq', 'fun': constraint, 'args': (center,)}
    res = minimize(objective, x0, method='SLSQP', constraints=[cons])

    if res.success:
        x_opt[t] = res.x
        x0 = res.x 

traj = np.array([trajectory(t) for t in range(T)])

plt.figure(figsize=(8,8))
plt.plot(traj[:,0], traj[:,1], 'ro-', label='Center of constraint')
plt.plot(x_opt[:,0], x_opt[:,1], 'bx--', label='Adaptative solution')
plt.legend()
plt.title("Adaptative following with explicit constraints (SLSQP)")
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.grid(True)
plt.savefig("opti_a_b.png")
