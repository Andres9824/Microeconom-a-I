import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

# Parámetros del sistema de Lorenz
sigma = 10
beta = 8/3
rho = 28

# Definimos las ecuaciones diferenciales
def lorenz_system(state, sigma, beta, rho):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

# Resolver la ecuación de Lorenz con múltiples trayectorias
def solve_lorenz_multiple(dt=0.01, num_steps=10000, num_streams=5, offset=5):
    trajectories = []
    for i in range(num_streams):
        state = np.array([1.0, 1.0 - i * offset * 0.1, 1.0 - i * offset * 0.2])
        trajectory = np.empty((num_steps, 3))
        
        for j in range(num_steps):
            trajectory[j] = state
            state += lorenz_system(state, sigma, beta, rho) * dt
        
        trajectories.append(trajectory)
    
    return trajectories

# Generar las trayectorias
data_streams = solve_lorenz_multiple()

# Configurar la figura de la animación
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-20, 20])
ax.set_ylim([-30, 30])
ax.set_zlim([0, 50])
ax.set_title("Simulación de Corrientes Atmosféricas en Forma de Tornado")

# Obtener colores de una escala de azules
colors = cm.Blues(np.linspace(0.3, 1, len(data_streams)))
lines = [ax.plot([], [], [], lw=0.5, color=colors[i])[0] for i in range(len(data_streams))]

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines

def update(frame):
    for line, data in zip(lines, data_streams):
        line.set_data(data[:frame, 0], data[:frame, 1])
        line.set_3d_properties(data[:frame, 2])
    return lines

anim = FuncAnimation(fig, update, frames=len(data_streams[0]), init_func=init, interval=10, blit=False)
plt.show()
