# MIT License
# Copyright (c) 2025 Santhosh S
# See LICENSE file for full license text.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numba
from numba import njit, prange
import os
import time
from pathlib import Path
from dataclasses import dataclass
import psutil
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for compatibility
plt.switch_backend('Agg')

@dataclass
class TurbulentConfig:
    """Configuration for turbulent flow simulation with dimensionless parameters."""
    # Physical parameters (dimensionless)
    L: float = 1.0  # Reference length (cylinder diameter)
    R_cylinder: float = 0.5  # Dimensionless radius
    V_inf: float = 1.0  # Reference velocity (dimensionless)
    cylinder_center: tuple[float, float] = (4.0, 2.0)  # Dimensionless coordinates

    # Domain parameters (dimensionless)
    x_min: float = 0.0
    x_max: float = 16.0
    y_min: float = 0.0
    y_max: float = 4.0

    # Mesh parameters
    nx: int = 400
    ny: int = 100

    # Time parameters (dimensionless)
    T_total: float = 30.0
    dt_initial: float = 1e-5  # Significantly smaller initial time step
    dt_min: float = 1e-6
    dt_max: float = 0.0005
    cfl_target: float = 0.05

    # Turbulence parameters
    Re: float = 80.0
    use_les: bool = True
    smagorinsky_constant: float = 0.17

    # Numerical stability
    use_supg: bool = True
    artificial_viscosity: float = 0.05
    pressure_relaxation: float = 0.9  # Increased for better pressure convergence
    max_velocity: float = 5.0
    initial_steps: int = 100  # Number of steps for gradual IBM application

    # Output parameters
    save_interval: int = 200
    output_dir: str = "turbulent_cylinder_results"
    dpi: int = 200
    memory_efficient: bool = True

    def __post_init__(self):
        self.dx = (self.x_max - self.x_min) / (self.nx - 1)
        self.dy = (self.y_max - self.y_min) / (self.ny - 1)
        self.nu = 1.0 / self.Re
        self.dt = self.dt_initial
        
        memory_mb = (self.nx * self.ny * 8 * 8) / (1024 * 1024)
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        print(f"--- Turbulent Flow Configuration (Dimensionless) ---")
        print(f"Reynolds Number: {self.Re}")
        print(f"Grid: {self.nx}x{self.ny}")
        print(f"Grid spacing: dx={self.dx:.4f}, dy={self.dy:.4f}")
        print(f"LES Model: {'Enabled' if self.use_les else 'Disabled'}")
        print(f"SUPG Stabilization: {'Enabled' if self.use_supg else 'Disabled'}")
        print(f"Estimated Memory: {memory_mb:.1f} MB")
        print(f"Available Memory: {available_mb:.1f} MB")
        if memory_mb > available_mb * 0.8:
            print("WARNING: High memory usage expected!")
        print("-----------------------------------\n")

@njit(parallel=True)
def compute_smagorinsky_viscosity(u, v, dx, dy, cs):
    """Compute Smagorinsky turbulent viscosity (dimensionless)."""
    ny, nx = u.shape
    nu_t = np.zeros_like(u)
    
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            dudx = (u[i, j+1] - u[i, j-1]) / (2 * dx)
            dudy = (u[i+1, j] - u[i-1, j]) / (2 * dy)
            dvdx = (v[i, j+1] - v[i, j-1]) / (2 * dx)
            dvdy = (v[i+1, j] - v[i-1, j]) / (2 * dy)
            
            S_mag = np.sqrt(2 * (dudx**2 + dvdy**2) + (dudy + dvdx)**2)
            delta = np.sqrt(dx * dy)
            nu_t[i, j] = (cs * delta)**2 * S_mag
            
    return nu_t

@njit(parallel=True)
def compute_supg_stabilization(u, v, dx, dy, dt, Pe_local):
    """Compute enhanced SUPG stabilization (dimensionless)."""
    ny, nx = u.shape
    tau_supg = np.zeros_like(u)
    
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            vel_mag = np.sqrt(u[i, j]**2 + v[i, j]**2)
            h_elem = min(dx, dy)
            
            if vel_mag > 1e-10:
                tau_supg[i, j] = h_elem / (2 * vel_mag) * min(1.0, Pe_local[i, j] / 2.0)
            else:
                tau_supg[i, j] = dt / 2
                
    return tau_supg

@njit(parallel=True, fastmath=True)
def compute_convection_supg(u, v, phi, dx, dy, tau_supg):
    """Compute convection with SUPG stabilization (dimensionless)."""
    ny, nx = phi.shape
    conv = np.zeros_like(phi)
    
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            dphidx = (phi[i, j+1] - phi[i, j-1]) / (2 * dx)
            dphidy = (phi[i+1, j] - phi[i-1, j]) / (2 * dy)
            conv_standard = u[i, j] * dphidx + v[i, j] * dphidy
            
            if tau_supg[i, j] > 0:
                d2phidx2 = (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) / dx**2
                d2phidy2 = (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]) / dy**2
                supg_term = tau_supg[i, j] * (u[i, j] * d2phidx2 + v[i, j] * d2phidy2)
                conv[i, j] = conv_standard - supg_term
            else:
                conv[i, j] = conv_standard
                
    return conv

class RobustTurbulentSolver:
    """2D Navier-Stokes solver for turbulent flows (dimensionless)."""
    
    def __init__(self, config: TurbulentConfig):
        self.config = config
        self.setup_grid()
        self.setup_boundary_masks()
        self.initialize_fields()
        self.setup_sparse_matrices()
        self.step = 0  # Track step for gradual IBM
        
    def setup_grid(self):
        """Initialize computational grid (dimensionless)."""
        cfg = self.config
        self.x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
        self.y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        
    def initialize_fields(self):
        """Initialize flow fields with memory-efficient arrays."""
        cfg = self.config
        dtype = np.float32 if cfg.memory_efficient else np.float64
        
        self.u = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
        self.v = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
        self.p = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
        self.nu_t = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
        self.tau_supg = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
        
        self.initialize_potential_flow()
        
    def initialize_potential_flow(self):
        """Initialize with smoothed potential flow (dimensionless)."""
        cfg = self.config
        x_c, y_c = cfg.cylinder_center
        for i in range(cfg.ny):
            for j in range(cfg.nx):
                x, y = self.X[i, j], self.Y[i, j]
                r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
                if r > cfg.R_cylinder + cfg.dx:  # Apply potential flow only outside buffer
                    theta = np.arctan2(y - y_c, x - x_c)
                    u_pot = 1.0 * (1 - (cfg.R_cylinder / r)**2 * np.cos(2 * theta))
                    v_pot = -1.0 * (cfg.R_cylinder / r)**2 * np.sin(2 * theta)
                    self.u[i, j] = u_pot * (1 - self.ibm_mask[i, j])
                    self.v[i, j] = v_pot * (1 - self.ibm_mask[i, j])
                else:
                    self.u[i, j] = 0
                    self.v[i, j] = 0
                    
    def setup_sparse_matrices(self):
        """Setup sparse Laplacian matrix with boundary conditions."""
        cfg = self.config
        n = cfg.nx * cfg.ny
        row_indices, col_indices, data = [],[],[]
        dx2_inv = 1.0 / cfg.dx**2
        dy2_inv = 1.0 / cfg.dy**2

        def add_entry(row, col, val):
            row_indices.append(row)
            col_indices.append(col)
            data.append(val)

        for i in range(cfg.ny):
            for j in range(cfg.nx):
                idx = i * cfg.nx + j
                if self.cylinder_mask[i, j]:
                    add_entry(idx, idx, 1.0)  # Fix pressure inside cylinder
                elif i == 0 and j == 0:
                    add_entry(idx, idx, 1.0)
                elif i == 0:
                    add_entry(idx, idx, -2 * dx2_inv - 2 * dy2_inv)
                    if j > 0:
                        add_entry(idx, idx - 1, dx2_inv)
                    if j < cfg.nx - 1:
                        add_entry(idx, idx + 1, dx2_inv)
                    add_entry(idx, idx + cfg.nx, 2 * dy2_inv)
                elif i == cfg.ny - 1:
                    add_entry(idx, idx, -2 * dx2_inv - 2 * dy2_inv)
                    if j > 0:
                        add_entry(idx, idx - 1, dx2_inv)
                    if j < cfg.nx - 1:
                        add_entry(idx, idx + 1, dx2_inv)
                    add_entry(idx, idx - cfg.nx, 2 * dy2_inv)
                elif j == 0:
                    add_entry(idx, idx, -2 * dy2_inv - 2 * dx2_inv)
                    add_entry(idx, idx - cfg.nx, dy2_inv)
                    add_entry(idx, idx + cfg.nx, dy2_inv)
                    add_entry(idx, idx + 1, 2 * dx2_inv)
                elif j == cfg.nx - 1:
                    add_entry(idx, idx, -2 * dy2_inv - 2 * dx2_inv)
                    add_entry(idx, idx - cfg.nx, dy2_inv)
                    add_entry(idx, idx + cfg.nx, dy2_inv)
                    add_entry(idx, idx - 1, 2 * dx2_inv)
                else:
                    add_entry(idx, idx, -2.0 * (dx2_inv + dy2_inv))
                    add_entry(idx, idx - cfg.nx, dy2_inv)
                    add_entry(idx, idx + cfg.nx, dy2_inv)
                    add_entry(idx, idx - 1, dx2_inv)
                    add_entry(idx, idx + 1, dx2_inv)

        self.laplacian_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    
    def setup_boundary_masks(self):
        """Setup masks with smoother IBM transition."""
        cfg = self.config
        x_c, y_c = cfg.cylinder_center
        dist = np.sqrt((self.X - x_c)**2 + (self.Y - y_c)**2)
        self.cylinder_mask = dist <= cfg.R_cylinder
        # Wider transition zone for IBM
        sigma = cfg.dx  # Increased from 0.5*dx
        self.ibm_mask = np.exp(-(dist - cfg.R_cylinder)**2 / sigma**2)
        self.ibm_mask = np.where(dist < cfg.R_cylinder, 1.0, 
                                 np.where(dist < cfg.R_cylinder + 3*cfg.dx, self.ibm_mask, 0.0))
    
    def adaptive_time_step(self):
        """Compute adaptive time step with stricter constraints."""
        cfg = self.config
        vel_max = max(np.max(np.abs(self.u)), np.max(np.abs(self.v)), 1e-10)
        dt_cfl = cfg.cfl_target * min(cfg.dx, cfg.dy) / vel_max
        nu_eff_max = cfg.nu + np.max(self.nu_t) + cfg.artificial_viscosity
        dt_visc = 0.25 * min(cfg.dx, cfg.dy)**2 / nu_eff_max
        self.config.dt = np.clip(min(dt_cfl, dt_visc), cfg.dt_min, cfg.dt_max)
        return self.config.dt
    
    def solve_pressure_poisson(self, div_u_star):
        """Solve pressure Poisson equation with cylinder boundary."""
        cfg = self.config
        rhs = div_u_star.flatten() / cfg.dt
        rhs[self.cylinder_mask.flatten()] = 0  # Zero RHS inside cylinder
        try:
            phi_flat = spsolve(self.laplacian_matrix, rhs)
            phi = phi_flat.reshape((cfg.ny, cfg.nx))
        except:
            phi = self.solve_pressure_iterative(div_u_star)
        phi[self.cylinder_mask] = 0  # Enforce zero pressure inside cylinder
        return phi
    
    def solve_pressure_iterative(self, div_u_star):
        """Iterative pressure solver with more iterations."""
        cfg = self.config
        phi = np.zeros_like(div_u_star)
        omega = 1.5
        for _ in range(500):  # Increased iterations
            phi_old = phi.copy()
            for i in range(1, cfg.ny-1):
                for j in range(1, cfg.nx-1):
                    if not self.cylinder_mask[i, j]:
                        rhs_val = div_u_star[i, j] / cfg.dt
                        phi_new = (cfg.dy**2 * (phi[i, j+1] + phi[i, j-1]) +
                                   cfg.dx**2 * (phi[i+1, j] + phi[i-1, j]) -
                                   rhs_val * cfg.dx**2 * cfg.dy**2) / (2 * (cfg.dx**2 + cfg.dy**2))
                        phi[i, j] = (1 - omega) * phi[i, j] + omega * phi_new
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = 0
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            phi[self.cylinder_mask] = 0
            if np.max(np.abs(phi - phi_old)) < 1e-7:  # Tighter convergence
                break
        return phi
    
    def apply_immersed_boundary(self):
        """Apply immersed boundary method with gradual forcing."""
        cfg = self.config
        # Ramp up IBM force over initial_steps
        force_strength = 100.0 * min(1.0, self.step / cfg.initial_steps)
        self.u = self.u * (1 - self.ibm_mask)
        self.v = self.v * (1 - self.ibm_mask)
        self.u -= force_strength * self.ibm_mask * self.u * cfg.dt
        self.v -= force_strength * self.ibm_mask * self.v * cfg.dt
    
    def clip_velocities(self):
        """Clip velocities to prevent instabilities."""
        cfg = self.config
        self.u = np.clip(self.u, -cfg.max_velocity, cfg.max_velocity)
        self.v = np.clip(self.v, -cfg.max_velocity, cfg.max_velocity)
    
    def time_step(self):
        """Perform one time step (dimensionless)."""
        cfg = self.config
        dt = self.adaptive_time_step()
        u_old = self.u.copy()
        v_old = self.v.copy()
        
        if cfg.use_les:
            self.nu_t = compute_smagorinsky_viscosity(self.u, self.v, cfg.dx, cfg.dy, cfg.smagorinsky_constant)
        
        if cfg.use_supg:
            Pe_local = np.sqrt(self.u**2 + self.v**2) * min(cfg.dx, cfg.dy) / (cfg.nu + self.nu_t + 1e-10)
            self.tau_supg = compute_supg_stabilization(self.u, self.v, cfg.dx, cfg.dy, dt, Pe_local)
        
        nu_eff = cfg.nu + self.nu_t + cfg.artificial_viscosity
        
        if cfg.use_supg:
            conv_u = compute_convection_supg(u_old, v_old, u_old, cfg.dx, cfg.dy, self.tau_supg)
            conv_v = compute_convection_supg(u_old, v_old, v_old, cfg.dx, cfg.dy, self.tau_supg)
        else:
            conv_u = np.zeros_like(u_old)
            conv_v = np.zeros_like(v_old)
            conv_u[1:-1, 1:-1] = (
                u_old[1:-1, 1:-1] * (u_old[1:-1, 2:] - u_old[1:-1, :-2]) / (2 * cfg.dx) +
                v_old[1:-1, 1:-1] * (u_old[2:, 1:-1] - u_old[:-2, 1:-1]) / (2 * cfg.dy)
            )
            conv_v[1:-1, 1:-1] = (
                u_old[1:-1, 1:-1] * (v_old[1:-1, 2:] - v_old[1:-1, :-2]) / (2 * cfg.dx) +
                v_old[1:-1, 1:-1] * (v_old[2:, 1:-1] - v_old[:-2, 1:-1]) / (2 * cfg.dy)
            )
        
        lap_u = np.zeros_like(u_old)
        lap_v = np.zeros_like(v_old)
        lap_u[1:-1, 1:-1] = (
            nu_eff[1:-1, 1:-1] * (
                (u_old[1:-1, 2:] - 2*u_old[1:-1, 1:-1] + u_old[1:-1, :-2]) / cfg.dx**2 +
                (u_old[2:, 1:-1] - 2*u_old[1:-1, 1:-1] + u_old[:-2, 1:-1]) / cfg.dy**2
            )
        )
        lap_v[1:-1, 1:-1] = (
            nu_eff[1:-1, 1:-1] * (
                (v_old[1:-1, 2:] - 2*v_old[1:-1, 1:-1] + v_old[1:-1, :-2]) / cfg.dx**2 +
                (v_old[2:, 1:-1] - 2*v_old[1:-1, 1:-1] + v_old[:-2, 1:-1]) / cfg.dy**2
            )
        )
        
        u_star = u_old + dt * (-conv_u + lap_u)
        v_star = v_old + dt * (-conv_v + lap_v)
        
        self.apply_boundary_conditions(u_star, v_star)
        u_star = u_star * (1 - self.ibm_mask)
        v_star = v_star * (1 - self.ibm_mask)
        
        div_u_star = np.zeros_like(u_star)
        div_u_star[1:-1, 1:-1] = (
            (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * cfg.dx) +
            (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * cfg.dy)
        )
        
        phi = self.solve_pressure_poisson(div_u_star)
        
        dpdx = np.zeros_like(phi)
        dpdy = np.zeros_like(phi)
        dpdx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * cfg.dx)
        dpdy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * cfg.dy)
        
        self.u = u_star - dt * dpdx
        self.v = v_star - dt * dpdy
        self.clip_velocities()
        
        self.p = cfg.pressure_relaxation * self.p + (1 - cfg.pressure_relaxation) * (self.p + phi)
        
        self.apply_boundary_conditions(self.u, self.v)
        self.apply_immersed_boundary()
        
        self.step += 1
    
    def apply_boundary_conditions(self, u, v):
        """Apply smoothed boundary conditions (dimensionless)."""
        cfg = self.config
        # Smoother perturbation
        perturbation = 0.02 * np.sin(2 * np.pi * self.y / cfg.y_max)  # Reduced amplitude
        u[:, 0] = 1.0 * (1 + perturbation)
        v[:, 0] = 0
        u[:, -1] = u[:, -2]  # Neumann condition at outflow
        v[:, -1] = v[:, -2]
        u[0, :] = 0  # No-slip at walls
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0
    
    def compute_vorticity(self):
        """Compute vorticity field (dimensionless)."""
        cfg = self.config
        vorticity = np.zeros_like(self.u)
        vorticity[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * cfg.dx) -
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * cfg.dy)
        )
        vorticity[self.cylinder_mask] = np.nan
        return vorticity

class TurbulentVisualizer:
    """Visualizer for velocity and vorticity fields with separate plots."""
    
    def __init__(self, config: TurbulentConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        plt.style.use('seaborn-v0_8')
        
    def plot_velocity_frame(self, solver: RobustTurbulentSolver, step: int, current_time: float):
        """Visualize velocity magnitude with streamlines and vectors."""
        cfg = self.config
        try:
            fig = plt.figure(figsize=(12, 6), facecolor='white')
            ax = fig.add_subplot(111)
            vel_mag = np.sqrt(solver.u**2 + solver.v**2)
            levels = np.linspace(0, np.nanmax(vel_mag)*0.9, 31)
            cf = ax.contourf(solver.X, solver.Y, vel_mag, levels=levels, cmap='viridis')
            plt.colorbar(cf, ax=ax, label='Dimensionless Velocity Magnitude |V|')
            try:
                skip = max(8, min(solver.X.shape) // 20)  # Adjusted for denser grid
                seed_points = np.array([
                    [cfg.x_min + 1, y] for y in np.linspace(cfg.y_min + 0.3, cfg.y_max - 0.3, 12)
                ])
                ax.streamplot(solver.X, solver.Y, solver.u, solver.v, 
                              color='white', linewidth=0.8, density=1.5,
                              start_points=seed_points, maxlength=50)
                # Add quiver plot for debugging
                ax.quiver(solver.X[::skip, ::skip], solver.Y[::skip, ::skip], 
                          solver.u[::skip, ::skip], solver.v[::skip, ::skip], 
                          color='lightgray', scale=20, alpha=0.5)
            except:
                pass
            cyl = patches.Circle(cfg.cylinder_center, cfg.R_cylinder,
                                 facecolor='black', edgecolor='gold', linewidth=1.5)
            ax.add_patch(cyl)
            ax.set_xlim(cfg.x_min, cfg.x_max)
            ax.set_ylim(cfg.y_min, cfg.y_max)
            ax.set_aspect('equal')
            ax.set_xlabel('x/L')
            ax.set_ylabel('y/L')
            ax.set_title(f'Velocity Field, Re={cfg.Re:.0f}, t={current_time:.2f}')
            ax.grid(True, alpha=0.3)
            
            max_vel = np.nanmax(vel_mag)
            mean_vel = np.nanmean(vel_mag)
            fig.text(0.02, 0.02, f'Max |V|: {max_vel:.3f} | Mean |V|: {mean_vel:.3f}', 
                     fontsize=8)
            
            plt.tight_layout()
            filename = self.output_path / f"velocity_frame_{step:06d}.png"
            plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight')
            plt.close(fig)
            
            if cfg.memory_efficient:
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"Error plotting velocity frame {step}: {e}")
            plt.close('all')
    
    def plot_vorticity_frame(self, solver: RobustTurbulentSolver, step: int, current_time: float):
        """Visualize vorticity field."""
        cfg = self.config
        try:
            fig = plt.figure(figsize=(12, 6), facecolor='white')
            ax = fig.add_subplot(111)
            vorticity = solver.compute_vorticity()
            vort_max = np.nanmax(np.abs(vorticity))
            levels = np.linspace(-vort_max*0.8, vort_max*0.8, 51)
            cf = ax.contourf(solver.X, solver.Y, vorticity, levels=levels, cmap='RdBu', extend='both')
            plt.colorbar(cf, ax=ax, label='Dimensionless Vorticity ω')
            cyl = patches.Circle(cfg.cylinder_center, cfg.R_cylinder,
                                 facecolor='black', edgecolor='gold', linewidth=1.5)
            ax.add_patch(cyl)
            ax.set_xlim(cfg.x_min, cfg.x_max)
            ax.set_ylim(cfg.y_min, cfg.y_max)
            ax.set_aspect('equal')
            ax.set_xlabel('x/L')
            ax.set_ylabel('y/L')
            ax.set_title(f'Vorticity Field, Re={cfg.Re:.0f}, t={current_time:.2f}')
            ax.grid(True, alpha=0.3)
            
            fig.text(0.02, 0.02, f'Vorticity Range: ±{vort_max:.2f}', fontsize=8)
            
            plt.tight_layout()
            filename = self.output_path / f"vorticity_frame_{step:06d}.png"
            plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight')
            plt.close(fig)
            
            if cfg.memory_efficient:
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"Error plotting vorticity frame {step}: {e}")
            plt.close('all')

def monitor_simulation_health(solver, step):
    """Monitor simulation stability with adaptive divergence threshold."""
    cfg = solver.config
    u_max = np.max(np.abs(solver.u))
    v_max = np.max(np.abs(solver.v))
    div_max = np.max(np.abs(
        (solver.u[1:-1, 2:] - solver.u[1:-1, :-2]) / (2 * cfg.dx) +
        (solver.v[2:, 1:-1] - solver.v[:-2, 1:-1]) / (2 * cfg.dy)
    ))
    
    if np.any(np.isnan(solver.u)) or np.any(np.isnan(solver.v)) or np.any(np.isnan(solver.p)):
        print(f"ERROR: NaN values detected at step {step}")
        return False
    
    if u_max > cfg.max_velocity or v_max > cfg.max_velocity:
        print(f"WARNING: High velocities detected at step {step}: u_max={u_max:.3f}, v_max={v_max:.3f}")
        return False
        
    # Relaxed divergence threshold for initial steps
    div_threshold = 5.0 if step > cfg.initial_steps else 10.0
    if div_max > div_threshold:
        print(f"WARNING: High divergence detected at step {step}: div_max={div_max:.3f}")
        return False
        
    return True

def main():
    """Main function for turbulent flow simulation."""
    config = TurbulentConfig(
        Re=80.0,
        nx=400,
        ny=100,
        T_total=30.0,
        use_les=True,
        use_supg=True,
        memory_efficient=True
    )
    
    print("Initializing Robust Turbulent CFD Solver (Dimensionless)...")
    solver = RobustTurbulentSolver(config)
    visualizer = TurbulentVisualizer(config)
    
    print(f"Starting turbulent simulation with Re={config.Re}")
    print(f"Grid resolution: {config.nx}x{config.ny}")
    print(f"LES model: {'Enabled' if config.use_les else 'Disabled'}")
    print(f"SUPG stabilization: {'Enabled' if config.use_supg else 'Disabled'}")
    
    start_time = time.time()
    step = 0
    current_time = 0.0
    
    try:
        while current_time < config.T_total:
            solver.time_step()
            current_time += config.dt
            
            if step % 20 == 0:
                if not monitor_simulation_health(solver, step):
                    print("Simulation became unstable. Stopping.")
                    break
                    
                wall_time = time.time() - start_time
                steps_per_sec = step / wall_time if wall_time > 0 else 0
                eta = (config.T_total - current_time) / config.dt / steps_per_sec if steps_per_sec > 0 else 0
                progress = (current_time / config.T_total) * 100
                print(f"Step: {step:6d} | Time: {current_time:6.2f} | dt: {config.dt:.2e} | "
                      f"Progress: {progress:.1f}% | Speed: {steps_per_sec:.1f} steps/s | ETA: {eta/60:.1f} min")
            
            if step % config.save_interval == 0:
                print(f"Saving frames at t={current_time:.2f}...")
                visualizer.plot_velocity_frame(solver, step, current_time)
                visualizer.plot_vorticity_frame(solver, step, current_time)
                if config.memory_efficient:
                    memory_usage = psutil.Process().memory_info().rss / (1024*1024)
                    print(f"  Memory usage: {memory_usage:.1f} MB")
            
            step += 1
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Simulation error: {e}")
    
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nSimulation completed:")
        print(f"  Total steps: {step}")
        print(f"  Final time: {current_time:.2f}")
        print(f"  Wall time: {total_time/60:.1f} minutes")
        print(f"  Average speed: {step/total_time:.1f} steps/second")
        print(f"  Results saved in: {visualizer.output_path}")

if __name__ == "__main__":
    main()