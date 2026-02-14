import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter

import main




# ========================= build the initial geometry ==========================

data = np.loadtxt("Planet300.dat")
data2 = np.copy(data)
#print(data.shape)


# --------------- set separation ----------------

R_body1 = np.max(np.linalg.norm(data[:,0:3], axis=1)) # max diamter of planet1
print("size of planet 1 = ", R_body1)

initial_separation = 5 * R_body1 # separation of the two planets
data2[:,0] += initial_separation # separate planet two's x-coords
data2[:,1] += R_body1 / 2 # make them not collide literally exactly head on (set planet2 y += fraction of R_body1)


# --------------- initial velocity ---------------

G = 6.6743e-11
M_total = np.sum(data[:,6])

v_impact = np.sqrt(2 * G * M_total / initial_separation)
print("impact velocity = ", v_impact)

# give initial and opposite v_x for both planets
data[:,3] += 1.5 * v_impact
data2[:,3] += -2.5 * v_impact


# =================== build SPHsystem (combined state vector) =======================

dim = 3
data_combined = np.vstack([data, data2])
N = data_combined.shape[0]
# print(N) # 602 particles, correct


r = data_combined[:,0:3]
v = data_combined[:,3:6]
m = data_combined[:,6]
rho = data_combined[:,7]
P = data_combined[:,8]


# initialize objects
kernel = main.cubicSplineKernel(dim, h = 5*1e6)
system = main.SPHsystem(N, dim, kernel)

gamma = system.gamma

system.S[:,0:3] = r
system.S[:,3:6] = v
system.S[:,-2] = rho
system.S[:,-1] = P / ((gamma - 1) * rho) # (e ??)
system.m = m 


# ---------- add spin to both planets ----------------
N_half = system.N // 2

T_spin = 4*1e3
T_spin2 = 8.5*1e3

main.add_spin(system, np.arange(0, N_half), T_spin)
main.add_spin(system, np.arange(N_half, system.N), T_spin2)



# ===================== solver =====================

"""
_______________________________
collision 0:
-----------------------
G = 6.6743*1e-11
h = 5*1e6
Nsteps = 2000
dt = 10
=> t[-1] = 20000

(No spin)

initial velocity +- v_impact/2 
(a bit too slow)

separeted 4*R_body1 in x only
_______________________________
collision 1:
-----------------------
G = 6.6743*1e-11
h = 5*1e6
Nsteps = 2000
dt = 10
=>t[-1] = 20000

T_spin = 8.5*1e3 (both)

initial velocity +- v_impact

separeted 3*R_body1 in x
R_body1 / 3 in y
_______________________________

collision 2:
-----------------------
G = 6.6743*1e-11
h = 5*1e6
Nsteps = 2000
dt = 10
=>t[-1] = 20000

T_spin = 1e3 planet1
T_spin = 1e2 planet2

initial velocity + 2 * v_impact (planet 1)
                 - 4 * v_impact (planet 2)

separeted 5*R_body1 in x
R_body1 / 2 in y

runtime warning: invalid value encoutered in sqrt (negative self energies)

wayyy to large spin lmao, palnets break before colliding
_______________________________


collision 3:
-----------------------
G = 6.6743*1e-11
h = 5*1e6
Nsteps = 2000
dt = 10
=>t[-1] = 20000

T_spin = 4e3 planet1
T_spin = 8.5e2 planet2

initial velocity + 1.5 * v_impact (planet 1)
                 - 2.5 * v_impact (planet 2)

separeted 5*R_body1 in x
R_body1 / 2 in y




_______________________________


"""

t0 = 0.0
dt = 10
Nsteps = 2000

times = np.linspace(0,dt*Nsteps,Nsteps)
print(times[-1])


S0 = system.S.flatten()
NS = main.NSequations(selfgrav_flag=True)


sol = solve_ivp(
    fun=lambda t, y: main.RHS(t, y, system, NS),
    t_span=(t0, dt*Nsteps),
    y0=S0,
    t_eval=times,
    method="RK45",
    max_step=dt,
    rtol=1e-4,
    atol=1e-7
)

# ================ plotting 3D ==========================

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')

# initial state from solver
S_init = sol.y[:,0].reshape(N, 2*dim + 2)
system.S[:] = S_init
system.density_summation()

rho_init = system.rho

sc = ax.scatter(
    S_init[:,0],
    S_init[:,1],
    S_init[:,2],
    c=rho_init,
    cmap='plasma',
    s=5
)

cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Density")

# set color scale from initial density
vmin = np.min(rho_init)
vmax = np.max(rho_init)
sc.set_clim(vmin, vmax)

# fix axes so they don’t rescale every frame
lim = 1.2 * np.max(np.abs(S_init[:,0:3]))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


# ------- update function ----------
def update(frame):
    S = sol.y[:,frame].reshape(N, 2*dim + 2)
    system.S[:] = S

    system.density_summation()
    rho = system.rho

    x = S[:,0]
    y = S[:,1]
    z = S[:,2]

    sc._offsets3d = (x, y, z)
    sc.set_array(rho)

    ax.set_title(f"t = {sol.t[frame]:.3f}")
    return sc,


# ---------- animation ----------
ani = FuncAnimation(
    fig,
    update,
    frames=len(sol.t),
    interval=50,
    blit=False
)

plt.show()
# writer = FFMpegWriter(fps=30)
# ani.save("./results/planet300_collision3.mp4", writer=writer)


