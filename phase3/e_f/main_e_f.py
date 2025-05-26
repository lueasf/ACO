import numpy as np
import math
import matplotlib.pyplot as plt 

T = 100  # steps
R_x, R_y = 4.0, 2.0
angles = np.linspace(0, 2*np.pi, T) 
c_path = np.column_stack((R_x * np.cos(angles), R_y * np.sin(angles)))
 
base_a, base_b = 1.5, 0.7
amp_a, amp_b = 0.3, 0.2
a_vals = base_a + amp_a * np.sin(2 * angles)
b_vals = base_b + amp_b * np.sin(3 * angles + 1.0)
phi_vals = 0.5 * angles
 
def ellipse_matrix(a, b, phi):
    cp, sp = math.cos(phi), math.sin(phi)
    R = np.array([[cp, -sp], [sp,  cp]])
    D = np.array([[1.0/(a**2), 0], [0, 1.0/(b**2)]])
    return R.dot(D).dot(R.T)
 
def compute_optimum_static(c, a, b, phi):
    A = ellipse_matrix(a, b, phi) 
    val = c.T.dot(A).dot(c) - 1  # = g(0) = (0-c)^T A (0-c) - 1
    if val <= 0: 
        return np.array([0.0, 0.0]) 
    def F_and_grad(lmbda): 
        M = np.linalg.inv(lmbda * A + np.eye(2)) 
        F_val = c.T.dot(M.T).dot(A).dot(M).dot(c) - 1.0 
        N = M.dot(A).dot(M)        # M A M
        F_grad = -2.0 * c.T.dot(N).dot(A).dot(M).dot(c)
        return float(F_val), float(F_grad) 
    lam = 1.0
    for _ in range(50):
        F_val, F_grad = F_and_grad(lam)
        if abs(F_val) < 1e-10:
            break
        lam -= F_val / F_grad
        if lam < 1e-8:  # garde λ positif
            lam = 1e-8 
    M = np.linalg.inv(lam * A + np.eye(2))
    d = -M.dot(c)
    return c + d
 
def log_barrier_newton(c, a, b, phi, x0, k=100.0, tol=1e-6, max_iters=50):
    A = ellipse_matrix(a, b, phi)
    def g_val(x): 
        return float((x - c).T.dot(A).dot(x - c) - 1.0)
    
    def phi_val(x):
        return -math.log(-g_val(x))
    x = x0.astype(float).copy()  # copie de x0 en flottant (éviter int)
    for it in range(max_iters):
        grad_f = 2.0 * x
        hess_f = 2.0 * np.eye(2)
        g = g_val(x)
        grad_g = 2.0 * A.dot(x - c)
        hess_g = 2.0 * A
        phi_grad = grad_g / (-g)                         
        phi_hess = hess_g / (-g) + np.outer(grad_g, grad_g) / (g**2)
        grad_L = k * grad_f + phi_grad
        hess_L = k * hess_f + phi_hess
        if np.linalg.norm(grad_L) < tol:
            break 
        dx = -np.linalg.solve(hess_L, grad_L) 
        alpha = 1.0
        x_new = x + alpha * dx 
        while g_val(x_new) >= 0 and alpha > 1e-8:
            alpha *= 0.5
            x_new = x + alpha * dx 
        L_curr = k * (x.dot(x)) + phi_val(x)
        L_new = k * (x_new.dot(x_new)) + phi_val(x_new)
        backtrack_count = 0
        while L_new > L_curr and backtrack_count < 10:
            alpha *= 0.5
            x_new = x + alpha * dx 
            if g_val(x_new) < 0:
                L_new = k * (x_new.dot(x_new)) + phi_val(x_new)
            backtrack_count += 1
        # Mise à jour de x
        x = x_new
    return x
 
x_opt = np.zeros((T, 2))     
x_barrier = np.zeros((T, 2)) 
 
x_prev = None
for t in range(T):
    c_t = c_path[t]
    a_t = a_vals[t]
    b_t = b_vals[t]
    phi_t = phi_vals[t]
    x_opt[t] = compute_optimum_static(c_t, a_t, b_t, phi_t)
    if t == 0 or x_prev is None:
        x0 = c_t.copy() 
    else:
        x0 = x_prev.copy()
        g_new = (x0 - c_t).T.dot(ellipse_matrix(a_t, b_t, phi_t)).dot(x0 - c_t) - 1.0
        if g_new >= 0:
            x0 = c_t.copy()
    x_sol = log_barrier_newton(c_t, a_t, b_t, phi_t, x0, k=100.0, tol=1e-6, max_iters=50)
    x_barrier[t] = x_sol
    x_prev = x_sol 
    
plt.figure(figsize=(6,6))
plt.plot(c_path[:,0], c_path[:,1], '-o', color='red', label='Centre des contraintes')
plt.plot(x_opt[:,0], x_opt[:,1], '-s', color='green', label='Optimum statique')
plt.plot(x_barrier[:,0], x_barrier[:,1], '-x', color='blue', label='Solution log-barrière')
plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
plt.title('Trajectoire dynamique: centre, optimum, solution adaptative')
plt.legend(); plt.grid(True); plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig('traj.png')

plt.figure(figsize=(6,4))
errors = np.linalg.norm(x_opt - x_barrier, axis=1)
plt.plot(range(T), errors, '-o', color='purple')
plt.xlabel('Temps $t$'); plt.ylabel('Erreur $||x^*(t) - x_b(t)||$')
plt.title('Erreur de suivi en fonction du temps')
plt.grid(True); plt.tight_layout()
plt.savefig('error_curve.png')