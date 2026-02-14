import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter

import main

# ========================= read .dat file ======================

data = np.loadtxt("Planet300.dat")
#print(data.shape)

x = data[:,0]
y = data[:,1]
z = data[:,2]

r = data[:, 0:3]
v = data[:, 3:6]
m = data[:, 6]
rho = data[:, 7]
P = data[:, 8]

# ======================= SPH system setup =================

dim = 3
N = data.shape[0] # 301

kernel = main.cubicSplineKernel(dim, h = 5*1e6)
system = main.SPHsystem(N, dim, kernel)

gamma = system.gamma

system.S[:,0:3] = r
system.S[:,3:6] = v
system.S[:,-2] = rho
system.S[:,-1] = P / ((gamma - 1) * rho) # (e ??)
system.m = m    

# ------------- add spin --------------
T_spin = 8.5e3
main.add_spin(system, np.arange(system.N), T_spin)


# ==================== solver =========================

"""
PARAMETER "SCAN":
___________________
Slow attraction (oscillating):
G = 10^{-11}
h = 1e7
Nsteps = 300
dt = 20
=> t[-1] = 6000
looks like gas cloud, doesnt pull together completely
___________________
Same as above but video lasts longer
G = 10^{-11}
h = 1e7
Nsteps = 800
dt = 20
=> t[-1] = 16000

__________________
G = 10^{-11}
h = 2e7
Nsteps = 800
dt = 20
=> t[-1] = 16000

diverges more before contracting (still gas-like)

_____________________
G = 10^{-11}
h = 5*1e6
Nsteps = 600
dt = 20
=> t[-1] = 12000

Fast inward contraction but outer shell of planets slower, looks a bit funky
(best sofar)
_____________________
G = 1e-11
h = 5*1e6
Nsteps = 2000
dt = 2
=>t[-1] = 4000

(added colormap for density here)
1:06 long video, looks cool at contracts and then oscillates a bit
becomes completely yellow when fully contracted... (i.e. max density on colormap)
____________________
(added spin here)

G = 1e-11
h = 5*1e6
Nsteps = 2000
dt = 10
=>t[-1] = 20000
T_spin = 8.5e6

similar to above, but spin is not noticeable
____________________
G = 1e-11
h = 5*1e6
Nsteps = 2000
dt = 10
=>t[-1] = 20000
T_spin = 8.5e3

(very succesfull video :) shows spinning and disk formation)
(best result sofar for single "planet/galaxy")
____________________
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
# ani.save("./results/planet300_testing.mp4", writer=writer)


