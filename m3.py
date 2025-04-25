import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

T   = 10 # steps
dx  = 0.4       
r   = 1.0 # radius of the pentagon
d   = np.array([1.0, 1.0])
d   /= np.linalg.norm(d)

angles = np.linspace(0, 2*np.pi, 6)[:-1] + np.pi/2  
V_rel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * r # shape (5,2)

x      = cp.Variable(2, name="x")
lam    = cp.Variable(5, nonneg=True, name="lambda")  
center = cp.Parameter(2, name="center")

constraints = [
    cp.sum(lam) == 1,
    x - center == V_rel.T @ lam
]

obj = cp.Maximize(d @ x)

prob = cp.Problem(obj, constraints)

trajectory = np.zeros((T,2))
centers    = np.zeros((T,2))

for t in range(T):
    c = np.array([t*dx, t*dx])
    center.value = c
    centers[t]  = c

    opt_val = prob.solve(solver=cp.ECOS, verbose=True)

    stats = prob.solver_stats
    trajectory[t] = x.value

    print(f"  Status      : {prob.status}")
    print(f"  Objet value : {opt_val:.6f}")

fig, ax = plt.subplots(figsize=(6,6))
for c in centers:
    verts = V_rel + c 
    poly = Polygon(verts, closed=True, facecolor="C0", edgecolor="k", alpha=0.2)
    ax.add_patch(poly)
 
ax.plot(trajectory[:,0], trajectory[:,1], 'o-', color="C1", lw=2, label="x*(t)")
ax.plot(*trajectory[-1], 's', color="C2", markersize=8, label="x*(T-1)")

ax.set_aspect('equal')
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.set_title('Optimization on a moving pentagon')
ax.legend()
ax.grid(True)
plt.show()
