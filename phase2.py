import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
 
# np.random.seed(43)

# Paramètres
n_steps      = 7      # nombre d'étapes
step_sigma   = 6.0      # écart-type du mouvement brownien
arrow_length = 2.5      # longueur de la flèche de direction
pause_time   = 0.1      # pause entre les frames (en s)
 
n_vertices    = 5
scale_factor  = 2.0  # Facteur d'échelle
angles        = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
base_vertices = scale_factor * np.column_stack((np.cos(angles), np.sin(angles)))

# État initial
pos         = np.array([0.0, 0.0])
orientation = np.random.uniform(0, 2*np.pi)

# Préparation de la figure
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))

# Tracés statiques initiaux
ax.set_aspect('equal')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_title("Optimization")
traj_line, = ax.plot([], [], '-k', lw=1, label="Trajectory")
opt_scat = ax.scatter([], [], c='C3', s=50, label="Optimal Points")
ax.legend(loc='upper left')

# Stockage pour mettre à jour
centers = []
all_vertices = []
directions = []
optimal_points = []

for step in range(n_steps): 

    # 1) Brownian translation
    pos = pos + np.random.normal(0, step_sigma, size=2)
    centers.append(pos.copy())
    
    # 2) Rotation aléatoire
    orientation = np.random.uniform(0, 2*np.pi)
    c, s = np.cos(orientation), np.sin(orientation)
    R = np.array([[c, -s], [s, c]])
    verts = base_vertices @ R + pos
    all_vertices.append(verts.copy())
    
    # 3) Direction aléatoire
    theta = np.random.uniform(0, 2*np.pi)
    d = np.array([np.cos(theta), np.sin(theta)])
    directions.append(d.copy())
    
    # 4) Optimisation Gurobi
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    lam = model.addVars(n_vertices, lb=0.0, name="lam")
    model.addConstr(gp.quicksum(lam[i] for i in range(n_vertices)) == 1)

    # 5) la fonction coût est la projection d'un point sur la direction d, c'est-à-dire le produit scalaire entre d et le point
    obj = gp.quicksum((d @ verts[i]) * lam[i] for i in range(n_vertices))
    
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    lam_vals = np.array([lam[i].X for i in range(n_vertices)])
    x_opt = (lam_vals[:, None] * verts).sum(axis=0)
    optimal_points.append(x_opt.copy())
    print(f"Step {step+1}: θ={theta:.2f} rad, optimal = ({x_opt[0]:.3f}, {x_opt[1]:.3f})")

    # --- Mise à jour de l'affichage cumulatif ---
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title(f"Step {step+1}/{n_steps}")

    # 1) Trajectoire
    centers_arr = np.array(centers)
    ax.plot(centers_arr[:,0], centers_arr[:,1], '-k', lw=1)

    # 2) Pentagones
    for verts_hist in all_vertices:
        closed = np.vstack([verts_hist, verts_hist[0]])
        ax.plot(closed[:,0], closed[:,1], '-', color='C0', alpha=0.5)

    # 3) Flèches
    for ctr, d_hist in zip(centers, directions):
        ax.arrow(ctr[0], ctr[1],
                 arrow_length*d_hist[0], arrow_length*d_hist[1],
                 head_width=0.15, head_length=0.2,
                 fc='C2', ec='C2', alpha=0.7)

    # 4) Points optimaux
    opt_arr = np.array(optimal_points)
    ax.scatter(opt_arr[:,0], opt_arr[:,1], c='C3', s=50)

    plt.draw()
    plt.pause(pause_time)

# À la fin, laisser fig figé
plt.ioff()
plt.show()
