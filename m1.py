import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def find_opti(direction):
    point = cp.Variable(2) # create the var to optimize
    contraintes = [cp.norm(point, 1) <= 1] # |x| + |y| <= 1
    probleme = cp.Problem(cp.Maximize(direction @ point), contraintes)
    probleme.solve(solver=cp.ECOS)
    
    if probleme.status != "optimal":
        raise ValueError("No solution found")
    
    return point.value

direction = np.array([1.0, 0.0])  
point_opt = find_opti(direction)

fig, ax = plt.subplots(figsize=(8, 8))


theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta) / (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
y = np.sin(theta) / (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
ax.fill(x, y, alpha=0.3, label="lozenge |x| + |y| ≤ 1")

ax.scatter(*point_opt, c='red', s=100, 
           label=f'Optimal Point ({point_opt[0]:.2f}, {point_opt[1]:.2f})')

ax.quiver(0, 0, direction[0], direction[1], 
          angles='xy', scale_units='xy', scale=1,
          color='green', width=0.01, label='Direction')

ax.set_title("Optimization on the lozenge")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.grid(True)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_aspect('equal')
ax.legend()
plt.show()