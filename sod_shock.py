import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter

import main


# =================== SOD SHOCK =======================
# ============ build Sod shock geometry ===============

dim = 1

# left region
Nx_L = 320
dx_L = 0.001875
x_L = -dx_L * np.arange(Nx_L-1,-1,-1) # arange magic

# right region
Nx_R = 80
dx_R = 0.0075 # particle separation
x_R = dx_R * np.arange(1, Nx_R + 1)

x = np.concatenate([x_L,x_R])
N = len(x)


# SPH sys
kernel = main.cubicSplineKernel(dim,h=0.01)
sys = main.SPHsystem(N, dim, kernel)


sys.m[:] = 0.001875
sys.S[:,0] = x
#print(sys.r)
sys.S[:,1] = 0.0
sys.density_summation()
sys.S[x<=0, 3] = 2.5    # left region energy
sys.S[x>0 , 3] = 1.795  # right region energy
print(sys.S)


# ============= solver ============

t0 = 0.0
dt = 0.005
Nsteps = 40

times = np.linspace(0,dt*Nsteps,Nsteps)
print(times)


S0 = sys.S.flatten()
NS = main.NSequations()


sol = solve_ivp(
    fun=lambda t, y: main.RHS(t, y, sys, NS),
    t_span=(t0, dt*Nsteps),
    y0=S0,
    t_eval=times,
    method="RK45",
    max_step=dt,
    rtol=1e-4,
    atol=1e-7
)


# ========================== static plots =======================
k = -1  # last (40:th) time step
S_flat_k = sol.y[:,k]

S_k = S_flat_k.reshape(N,4)
sys.S[:] = S_k
sys.density_summation()

x_k = sys.r
v_k = sys.v
rho_k = sys.rho
e_k = sys.e
P_k = sys.pressure()

plt.plot(x_k, rho_k)
plt.xlim((-0.4,0.4))
plt.xlabel("x [m]")
plt.ylabel("Density [kg/m³]")
plt.grid()
plt.show()

plt.plot(x_k, P_k)
plt.xlim((-0.4,0.4))
plt.xlabel("x [m]")
plt.ylabel("Pressure [N/m²]")
plt.grid()
plt.show()

plt.plot(x_k, v_k)
plt.xlabel("x [m]")
plt.ylabel("Velocity [m/s]")
plt.xlim((-0.4,0.4))
plt.grid()
plt.show()

plt.plot(x_k, e_k)
plt.xlabel("x [m]")
plt.ylabel("Internal energy [J/kg]")
plt.xlim((-0.4,0.4))
plt.grid()
plt.show()


fig, axs = plt.subplots(4,1,figsize=(8,8))
(ax_v, ax_rho, ax_p, ax_e) = axs

scat_v   = ax_v.scatter(x_k, v_k, s=12)
scat_rho = ax_rho.scatter(x_k, rho_k, s=12)
scat_p   = ax_p.scatter(x_k, P_k, s=12)
scat_e   = ax_e.scatter(x_k, e_k, s=12)


for ax in axs.flat:
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(0, 1.2)
    ax.grid()

axs.flat[-1].set_ylim(1,2.8)

ax_v.set_ylabel("Velocity [m/s]")
ax_rho.set_ylabel("Density [kg/m³]")
ax_p.set_ylabel("Pressure [N/m²]")
ax_e.set_ylabel("Internal energy [J/kg]")

ax_e.set_xlabel("x [m]")
plt.show()


# ====================== Sod shock simulation =================

fig, axs = plt.subplots(4,1,figsize=(8,8))
(ax_v, ax_rho, ax_p, ax_e) = axs

scat_v   = ax_v.scatter([], [], s=12)
scat_rho = ax_rho.scatter([], [], s=12)
scat_p   = ax_p.scatter([], [], s=12)
scat_e   = ax_e.scatter([], [], s=12)


for ax in axs.flat:
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(0, 1.2)
    ax.grid()

axs.flat[-1].set_ylim(1,2.8)

ax_v.set_ylabel("Velocity [m/s]")
ax_rho.set_ylabel("Density [kg/m³]")
ax_p.set_ylabel("Pressure [N/m²]")
ax_e.set_ylabel("Internal energy [J/kg]")

ax_e.set_xlabel("x [m]")



def update(frame):
    S = sol.y[:, frame].reshape(N, 4)

    sys.S[:] = S
    sys.density_summation()

    x   = sys.S[:,0]
    v   = sys.S[:,1]
    rho = sys.rho
    e   = sys.e
    P   = sys.pressure()

    scat_v.set_offsets(np.column_stack((x, v)))
    scat_rho.set_offsets(np.column_stack((x, rho)))
    scat_p.set_offsets(np.column_stack((x, P)))
    scat_e.set_offsets(np.column_stack((x, e)))

    fig.suptitle(f"Sod shock tube — t = {sol.t[frame]:.4f}")

    return scat_v, scat_rho, scat_p, scat_e


ani = FuncAnimation(
    fig,
    update,
    frames=len(sol.t),
    interval=50,
    blit=True
)


plt.show()
# writer = FFMpegWriter(fps=10, bitrate=1800)
# ani.save("./results/sod_all_fields_vs_x.mp4", writer=writer)




