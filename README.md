# SPH for Planet Collisions
Welcome! This repository contains my DEs-in-Astrophysics project in the course "FYSN33 Applied Computational Physics and Machine Learning" given at Lund University during the Spring semester of 2026. It contains numerical implementation and solutions to the Navier-Stokes (NS) equations in a Smoothed-particle-hydrodynamics (SPH) framework. The SPH implementation is centered around use of a state-vector for particles on a lattice. It is tested against the Sod-shock problem in 1D and simulates (in 3D) massive, gravitationally interacting particles. 


## What's in this repo
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
- `results/`
  - List of figures and videos showing different simulation runs


## Sod shock


https://github.com/user-attachments/assets/49a245ca-98e8-4e27-a4ae-007700c704f5



## Planet Collisions


https://github.com/user-attachments/assets/383f91ad-89c5-47f0-be64-9fd2fb7350bf


https://github.com/user-attachments/assets/b9351418-5ae9-4413-ab18-db528137227f


https://github.com/user-attachments/assets/b2e80930-ae5d-413b-90be-0d5a4a00505f


https://github.com/user-attachments/assets/66712bf8-5531-4936-b6e4-604a8e8a287b

