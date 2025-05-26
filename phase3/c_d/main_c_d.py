import numpy as np
import math
import matplotlib.pyplot as plt

"""
Minimize L(x) = k*f0(x) + phi(x; C) for a disk C
- center : center of the disk C
- x0 : starting point 
- k : log-barrière
- R : disk radius
return interior point x that minimizes L
"""
def solve_barrier(center, x0, k, R, max_iters=20, tol=1e-6):
    x = np.array(x0, dtype=float)
    cx, cy = center
    for iteration in range(max_iters): 
        dx = x[0] - cx
        dy = x[1] - cy
        D = R*R - (dx**2 + dy**2)
        if D <= 0:
            D = 1e-12
        grad_f0 = np.array([-1.0, -1.0])
        grad_phi = np.array([2*dx/D, 2*dy/D])
        H11 = 2.0/D + 4.0*dx*dx/(D*D)
        H22 = 2.0/D + 4.0*dy*dy/(D*D)
        H12 = 4.0*dx*dy/(D*D)
        H = np.array([[H11, H12],
                      [H12, H22]])
        grad_L = k * grad_f0 + grad_phi
        delta = -np.linalg.solve(H, grad_L)
        if np.linalg.norm(delta) < tol:
            break
        u = np.array([dx, dy])
        v = delta
        a = np.dot(v, v)
        b = 2*np.dot(u, v)
        c = np.dot(u, u) - R*R
        alpha_max = 1.0
        if a != 0:
            disc = b*b - 4*a*c
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                alpha1 = (-b + sqrt_disc) / (2*a)
                alpha2 = (-b - sqrt_disc) / (2*a)
                alphas = [alpha for alpha in (alpha1, alpha2) if alpha > 0]
                if alphas:
                    alpha_max = min(alphas)
        alpha = 0.95 * alpha_max
        L_current = k * (x[0] + x[1]) * -1 + (-math.log(D))   # -k*(x1+x2) - log(D)
        x_try = x + alpha * delta
        dx_try = x_try[0] - cx
        dy_try = x_try[1] - cy
        D_try = R*R - (dx_try**2 + dy_try**2)
        if D_try <= 0:  # on évite log(0)
            D_try = 1e-12
        L_try = -k*(x_try[0] + x_try[1]) - math.log(D_try)

        while L_try > L_current and alpha > 1e-6:
            alpha *= 0.5
            x_try = x + alpha * delta
            dx_try = x_try[0] - cx
            dy_try = x_try[1] - cy
            D_try = R*R - (dx_try**2 + dy_try**2)
            if D_try <= 0:
                D_try = 1e-12
            L_try = -k*(x_try[0] + x_try[1]) - math.log(D_try)
        x = x_try
        if np.linalg.norm(grad_L) < tol:
            break
    return x

T = 21
R = 2.0
k_barriere = 10

start_center = np.array([0.0, 0.0])
end_center   = np.array([4.0, 2.0])
centers = [ (1 - t/(T-1)) * start_center + (t/(T-1)) * 
           end_center  for t in range(T) ]
centers = np.array(centers)

optimum_true = []
for c in centers:
    direction = np.array([1.0, 1.0]) / np.sqrt(2)
    x_star = c + R * direction
    optimum_true.append(x_star)
optimum_true = np.array(optimum_true)

# Suivi adaptatif par log-barrière
solutions = []
x_t = solve_barrier(centers[0], x0=centers[0], k=k_barriere, R=R)
solutions.append(x_t)
for t in range(1, T):
    c_prev = centers[t-1]
    c_curr = centers[t]
    x_prev = solutions[-1]
    dist2 = np.linalg.norm(x_prev - c_curr)**2
    if dist2 >= R*R:
        direction_to_center = (x_prev - c_curr) / math.sqrt(dist2)
        x_start = c_curr + 0.9 * R * direction_to_center
    else:
        x_start = x_prev
    x_t = solve_barrier(c_curr, x_start, k=k_barriere, R=R)
    solutions.append(x_t)
solutions = np.array(solutions)

t = np.arange(len(centers))

plt.figure(figsize=(8, 6))
plt.plot(centers[:, 0],     centers[:, 1],     'ro-', label='Center of constraint')
plt.plot(optimum_true[:, 0], optimum_true[:, 1], 'g*--', label='Optimum statique')
plt.plot(solutions[:, 0],   solutions[:, 1],   'b^-.', label='Adaptative Log-barrier')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Comparison : center, static optimum and log-barrier method")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.savefig("comp_c_d2.png")