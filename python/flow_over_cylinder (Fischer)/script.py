# flow_over_cylinder_enhanced.py
#
# A 2D CFD simulation of flow past a cylinder using a projection method
# based on the dimensionless Navier-Stokes equations.
# This script is designed for clarity, robustness, and high-quality visualization.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# =============================================================================
# PART 1: SIMULATION PARAMETERS (THE "CONTROL PANEL")
# =============================================================================
# -- 1.1 Physics Parameters --
Re = 80.0  # Reynolds number: The sole physical parameter.

# -- 1.2 Domain and Grid Parameters --
Lx_star = 32.0  # Dimensionless domain length (in cylinder diameters)
Ly_star = 16.0  # Dimensionless domain height (in cylinder diameters)
Nx = 640        # Number of grid points in x
Ny = 320        # Number of grid points in y
dx_star = Lx_star / Nx  # Dimensionless grid spacing in x
dy_star = Ly_star / Ny  # Dimensionless grid spacing in y

# -- 1.3 Cylinder Parameters --
D_star = 1.0      # Dimensionless cylinder diameter (is always 1)
Cx_star = Lx_star / 4.0  # Cylinder center x-position
Cy_star = Ly_star / 2.0  # Cylinder center y-position
Radius_star = D_star / 2.0

# -- 1.4 Time and Stability Parameters --
CFL = 0.1       # Courant number for stability
T_star = 200.0  # Total dimensionless simulation time

# Time step is calculated based on CFL condition, not chosen arbitrarily
# Characteristic velocity is U_inf, which is 1 in dimensionless units.
dt_star = CFL * min(dx_star, dy_star) / 1.0
Nt = int(T_star / dt_star)  # Number of time steps

# -- 1.5 Solver Parameters --
P_SOLVER_ITERATIONS = 50  # Number of iterations for the pressure Poisson solver

# -- 1.6 Visualization Parameters --
PLOT_INTERVAL = 500  # Create a plot every N steps

# =============================================================================
# PART 2: GRID AND INITIALIZATION
# =============================================================================
print("--- Simulation Parameters ---")
print(f"Reynolds Number (Re): {Re}")
print(f"Domain Size (Lx*, Ly*): {Lx_star}, {Ly_star}")
print(f"Grid Resolution (Nx, Ny): {Nx}, {Ny}")
print(f"Dimensionless Time Step (dt*): {dt_star:.4e}")
print(f"Total Time Steps (Nt): {Nt}")
print("-----------------------------")

# Create meshgrid
X_star, Y_star = np.meshgrid(np.linspace(0, Lx_star, Nx), np.linspace(0, Ly_star, Ny))

# Initialize velocity fields (u*, v*) and pressure field (p*)
u_star = np.ones((Ny, Nx))  # Start with uniform flow
v_star = np.zeros((Ny, Nx))
p_star = np.zeros((Ny, Nx))

# Create a cylinder mask
cylinder_mask = (X_star - Cx_star)**2 + (Y_star - Cy_star)**2 < Radius_star**2

# =============================================================================
# PART 3: THE MAIN SOLVER LOOP
# =============================================================================

for n in range(Nt):
    # --- 3.1 Store previous velocity ---
    u_prev = u_star.copy()
    v_prev = v_star.copy()

    # --- 3.2 Advection-Diffusion Step (Provisional Velocity) ---
    # Central differences for diffusion (Laplacian)
    laplacian_u = (np.roll(u_prev, 1, axis=1) + np.roll(u_prev, -1, axis=1) +
                   np.roll(u_prev, 1, axis=0) + np.roll(u_prev, -1, axis=0) - 4 * u_prev) / (dx_star**2)
    laplacian_v = (np.roll(v_prev, 1, axis=1) + np.roll(v_prev, -1, axis=1) +
                   np.roll(v_prev, 1, axis=0) + np.roll(v_prev, -1, axis=0) - 4 * v_prev) / (dy_star**2)

    # Upwind differences for advection
    dudx = np.where(u_prev > 0, (u_prev - np.roll(u_prev, 1, axis=1)) / dx_star, (np.roll(u_prev, -1, axis=1) - u_prev) / dx_star)
    dudy = np.where(v_prev > 0, (u_prev - np.roll(u_prev, 1, axis=0)) / dy_star, (np.roll(u_prev, -1, axis=0) - u_prev) / dy_star)
    dvdx = np.where(u_prev > 0, (v_prev - np.roll(v_prev, 1, axis=1)) / dx_star, (np.roll(v_prev, -1, axis=1) - v_prev) / dx_star)
    dvdy = np.where(v_prev > 0, (v_prev - np.roll(v_prev, 1, axis=0)) / dy_star, (np.roll(v_prev, -1, axis=0) - v_prev) / dy_star)

    advection_u = u_prev * dudx + v_prev * dudy
    advection_v = u_prev * dvdx + v_prev * dvdy

    # Calculate provisional velocity (u_star_prov, v_star_prov)
    u_star_prov = u_prev - dt_star * advection_u + (dt_star / Re) * laplacian_u
    v_star_prov = v_prev - dt_star * advection_v + (dt_star / Re) * laplacian_v

    # --- 3.3 Pressure-Poisson Step ---
    # Calculate divergence of the provisional velocity field
    div_prov = ((np.roll(u_star_prov, -1, axis=1) - np.roll(u_star_prov, 1, axis=1)) / (2 * dx_star) +
                (np.roll(v_star_prov, -1, axis=0) - np.roll(v_star_prov, 1, axis=0)) / (2 * dy_star))

    # Right-hand side of the Poisson equation
    rhs_poisson = div_prov / dt_star

    # Solve the Poisson equation for pressure using Jacobi iteration
    p_next = p_star.copy()
    for _ in range(P_SOLVER_ITERATIONS):
        p_temp = p_next.copy()
        p_next[1:-1, 1:-1] = (((p_temp[1:-1, 2:] + p_temp[1:-1, :-2]) * dy_star**2 +
                               (p_temp[2:, 1:-1] + p_temp[:-2, 1:-1]) * dx_star**2 -
                               rhs_poisson[1:-1, 1:-1] * dx_star**2 * dy_star**2) /
                              (2 * (dx_star**2 + dy_star**2)))
        
        # Pressure boundary conditions
        p_next[:, -1] = 0  # Outlet pressure is zero
        p_next[:, 0] = p_next[:, 1]  # Inlet zero-gradient
        p_next[0, :] = p_next[1, :]  # Top wall zero-gradient
        p_next[-1, :] = p_next[-2, :]  # Bottom wall zero-gradient

    p_star = p_next

    # --- 3.4 Correction Step ---
    # Calculate pressure gradient
    dpdx = (np.roll(p_star, -1, axis=1) - np.roll(p_star, 1, axis=1)) / (2 * dx_star)
    dpdy = (np.roll(p_star, -1, axis=0) - np.roll(p_star, 1, axis=0)) / (2 * dy_star)

    # Update velocity to be divergence-free
    u_star = u_star_prov - dt_star * dpdx
    v_star = v_star_prov - dt_star * dpdy

    # --- 3.5 Apply Boundary Conditions ---
    # Inlet
    u_star[:, 0] = 1.0
    v_star[:, 0] = 0.0
    # Outlet (zero gradient)
    u_star[:, -1] = u_star[:, -2]
    v_star[:, -1] = v_star[:, -2]
    # Top wall (free slip)
    u_star[0, :] = u_star[1, :]
    v_star[0, :] = 0.0
    # Bottom wall (free slip)
    u_star[-1, :] = u_star[-2, :]
    v_star[-1, :] = 0.0
    # Cylinder (no-slip)
    u_star[cylinder_mask] = 0.0
    v_star[cylinder_mask] = 0.0

    # --- 3.6 Visualization and Progress ---
    if n % PLOT_INTERVAL == 0:
        print(f"Step: {n}/{Nt}, Dimensionless Time: {n * dt_star:.2f}/{T_star:.2f}")

        # --- Velocity Plot ---
        plt.figure(figsize=(16, 8))
        velocity_magnitude = np.sqrt(u_star**2 + v_star**2)
        plt.contourf(X_star, Y_star, velocity_magnitude, levels=100, cmap='plasma')
        plt.colorbar(label='Dimensionless Velocity Magnitude $|V^*|$')
        
        # Streamlines
        # Downsample for cleaner streamline plot
        skip = 8
        plt.streamplot(X_star[::skip, ::skip], Y_star[::skip, ::skip],
                       u_star[::skip, ::skip], v_star[::skip, ::skip],
                       color='black', linewidth=0.7, density=2)

        # Cylinder outline
        cylinder_plot = plt.Circle((Cx_star, Cy_star), Radius_star, color='black', fill=True)
        plt.gca().add_artist(cylinder_plot)
        
        plt.title(f'Velocity Magnitude and Streamlines at Re={Re}, t*={n * dt_star:.2f}')
        plt.xlabel('x/D')
        plt.ylabel('y/D')
        plt.axis('equal')
        plt.savefig(f'velocity_Re{Re}_step{n}.png', dpi=300)
        plt.close()

        # --- Vorticity Plot ---
        plt.figure(figsize=(16, 8))
        dvdx = (np.roll(v_star, -1, axis=1) - np.roll(v_star, 1, axis=1)) / (2 * dx_star)
        dudy = (np.roll(u_star, -1, axis=0) - np.roll(u_star, 1, axis=0)) / (2 * dy_star)
        vorticity = dvdx - dudy
        vorticity[cylinder_mask] = 0 # Mask vorticity inside cylinder

        vort_max = np.max(np.abs(vorticity))
        vort_limit = max(vort_max, 0.1) # Avoid zero limit at start

        plt.contourf(X_star, Y_star, vorticity, levels=100, cmap='RdBu',
                     vmin=-vort_limit, vmax=vort_limit)
        plt.colorbar(label='Dimensionless Vorticity $\omega^*$')

        # Cylinder outline
        cylinder_plot = plt.Circle((Cx_star, Cy_star), Radius_star, color='black', fill=True)
        plt.gca().add_artist(cylinder_plot)

        plt.title(f'Vorticity Field at Re={Re}, t*={n * dt_star:.2f}')
        plt.xlabel('x/D')
        plt.ylabel('y/D')
        plt.axis('equal')
        plt.savefig(f'vorticity_Re{Re}_step{n}.png', dpi=300)
        plt.close()

print("Simulation Finished.")