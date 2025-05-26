import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def find_opti(direction):
    point = cp.Variable(2)
    
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    x_verts = np.cos(angles)
    y_verts = np.sin(angles)
     
    constraints = []
    for i in range(5):
        x1, y1 = x_verts[i], y_verts[i]
        x2, y2 = x_verts[(i+1)%5], y_verts[(i+1)%5]
         
        a = y2 - y1
        b = x1 - x2
        c = x1*y2 - x2*y1
        constraints.append(a*point[0] + b*point[1] <= c)
     
    problem = cp.Problem(cp.Maximize(direction @ point), constraints)
    problem.solve(solver=cp.ECOS)
     
    return point.value
 
direction = np.array([1.0, 2.0]) 
point_opt = find_opti(direction)
 
 
fig, ax = plt.subplots(figsize=(8, 8))
angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
x_pent = np.cos(angles)
y_pent = np.sin(angles)
ax.fill(x_pent, y_pent, alpha=0.3)
 
ax.scatter(*point_opt, c='red', s=100, 
           label=f'Point optimal ({point_opt[0]:.2f}, {point_opt[1]:.2f})')
 
ax.quiver(0, 0, direction[0], direction[1], 
          angles='xy', scale_units='xy', scale=1,
          color='green', width=0.01, label='Direction')

ax.set_title("Optimization on pentagon")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.grid(True)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_aspect('equal')
ax.legend()
plt.show()