# SPH for Planet Collisions
Welcome! This repository contains my DEs-in-Astrophysics project in the course "FYSN33 Applied Computational Physics and Machine Learning" given at Lund University during the Spring semester of 2026. It contains numerical implementation and solutions to the Navier-Stokes (NS) equations in a Smoothed-particle-hydrodynamics (SPH) framework. The SPH implementation is centered around use of a state-vector for particles on a lattice. It is tested against the Sod-shock problem in 1D and simulates (in 3D) massive, gravitationally interacting particles. 


## :clipboard: What's in this repo
- `docs/`
  - `Notes.ipynb`: contains detailed mathematical motivation for the SPH framework.
  - `Report_Project_1.pdf`: written report focusing on implementation details. 
- `main.py`: central computational core:
  - classes `SPHsystem`, `cubicSplineKernel`, and `NSequations`.
  - helper functions `RHS()` and `add_spin()`.
- `sod_shock.py`:
  - Initialization and simulation of the 1D Sod shock problem.
- `planet_collision.py`:
  - Initialization and simulation of colliding planets.
- `planets/`
  - Comma-separated `.txt` data files containing initialization data: $x$, $y$, $z$, $v_x$, $v_y$, $v_z$, $m$, $\rho$, $p$, of 300, 600, 1200, and 2400 particles respectively.
- `results/`
  - List of figures and videos showing different simulation runs.


## Quick commands
```bash
commands to installl requirements
```


## :blue_book: Background on SPH
The Smoothed Particle Hydrodynamic (SPH) framework is defined by a system of point-like particles which interact with one another through so-called integration kernels, or smoothing functions. 


## :mag: Implementation details


## Results
### :dart: Sod shock


https://github.com/user-attachments/assets/49a245ca-98e8-4e27-a4ae-007700c704f5



### :earth_africa: Planet Collisions
Below are a few selected simulations of "planets" colliding. Clumps of particles labelled `planet_300` are each comprised of 300 state-vector elements (particles/points) with psotition, velocity, mass, density and self-pressure entries. These are initialized as state-vector elements and simulated using self-gravity and in some cases spin.

Video 0: One planet of 300 particles with self-gravity and initial spin. Notice that disc-formation naturally arrises. 

https://github.com/user-attachments/assets/383f91ad-89c5-47f0-be64-9fd2fb7350bf

Video 1: Two `planet_300` initially separated only in $x$-coordinates, impinging on one another with a static initial velocity $v_x$. Self-gravity causes planet formation, mutual acceleration and eventually stabilization towards a single planet-like ball.

https://github.com/user-attachments/assets/b9351418-5ae9-4413-ab18-db528137227f

Video 2: Two `planet_300` impining on one another separated in $x$ with a small $y$-offset. Initial spin $\omega$ and linear velocity $v_x$. Disc-formation is also visible, in particular after the collision.

https://github.com/user-attachments/assets/b2e80930-ae5d-413b-90be-0d5a4a00505f

Video 3: Different initial spins and larger initial linear velocity. Notice that the planet with large spin does not stabilize before impact, and the collission results in two planets of different volume diverging away from one another. 

https://github.com/user-attachments/assets/66712bf8-5531-4936-b6e4-604a8e8a287b

