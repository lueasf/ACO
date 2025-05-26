import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GRID_RES = 100
X_RANGE = (-3, 3)
Y_RANGE = (-3, 3)

class ConstraintSets:
    @staticmethod
    def pentagon(center, size, rotation=0): 
        angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + rotation
        return np.array([
            center[0] + size*np.cos(angles),
            center[1] + size*np.sin(angles)
        ])

    @staticmethod
    def rotating_square(center, size, angle): 
        return np.array([
            center[0] + size*np.array([np.cos(angle), -np.sin(angle), -np.cos(angle), np.sin(angle)]),
            center[1] + size*np.array([np.sin(angle), np.cos(angle), -np.sin(angle), -np.cos(angle)])
        ])

    @staticmethod
    def circle(center, radius): 
        theta = np.linspace(0, 2*np.pi, 100)
        return np.array([
            center[0] + radius*np.cos(theta),
            center[1] + radius*np.sin(theta)
        ])
 
def linear_cost(x, direction):
    return np.dot(x, direction)

def quadratic_cost(x, target):
    return np.linalg.norm(x - target)**2
 
class OptimizationScenario:
    def __init__(self): 
        self.x = np.linspace(*X_RANGE, GRID_RES)
        self.y = np.linspace(*Y_RANGE, GRID_RES)
        self.X, self.Y = np.meshgrid(self.x, self.y)
         
        self.timesteps = 50
        self.current_step = 0
         
        self.constraint_centers = np.array([
            [0.5*np.cos(2*np.pi*t/self.timesteps), 
             0.5*np.sin(2*np.pi*t/self.timesteps)] 
            for t in range(self.timesteps)
        ])
         
        self.target_positions = np.array([
            [2*np.cos(2*np.pi*t/self.timesteps), 
             2*np.sin(2*np.pi*t/self.timesteps)] 
            for t in range(self.timesteps)
        ])

    def get_constraints(self, t): 
        return ConstraintSets.pentagon(
            center=self.constraint_centers[t],
            size=0.8,
            rotation=0.1*t
        )

    def is_feasible(self, x, y, vertices): 
        from matplotlib.path import Path
        return Path(vertices.T).contains_point((x, y))

    def find_optimal(self, t): 
        vertices = self.get_constraints(t)
        target = self.target_positions[t]
        
        min_cost = float('inf')
        optimal_point = None
         
        for i in range(GRID_RES):
            for j in range(GRID_RES):
                x, y = self.X[i,j], self.Y[i,j]
                if self.is_feasible(x, y, vertices):
                    current_cost = quadratic_cost([x,y], target)
                    if current_cost < min_cost:
                        min_cost = current_cost
                        optimal_point = [x, y]
        
        return optimal_point, vertices, target

scenario = OptimizationScenario()

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(*X_RANGE)
ax.set_ylim(*Y_RANGE)

constraint_patch = plt.Polygon([[0,0]], alpha=0.2, color='blue')
target_point = plt.Circle((0,0), 0.1, color='green')
optimal_point = plt.Circle((0,0), 0.1, color='red')
ax.add_patch(constraint_patch)
ax.add_patch(target_point)
ax.add_patch(optimal_point)

def update(frame):
    t = frame % scenario.timesteps
    point, vertices, target = scenario.find_optimal(t)
    
    constraint_patch.set_xy(vertices.T)
    target_point.center = target
    optimal_point.center = point
    
    ax.set_title(f"Timestep {t+1}/{scenario.timesteps}\n"
                 f"Target: {target.round(2)} | Optimal: {np.array(point).round(2)}")
    
    return constraint_patch, target_point, optimal_point

ani = FuncAnimation(fig, update, frames=scenario.timesteps, interval=200, blit=True)
plt.show()