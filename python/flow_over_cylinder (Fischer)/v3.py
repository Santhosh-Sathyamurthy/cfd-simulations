import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import njit, prange
import os
import time
from pathlib import Path
from dataclasses import dataclass
import psutil
import warnings
from concurrent.futures import ThreadPoolExecutor
import gc
warnings.filterwarnings('ignore')

# Set matplotlib backend for compatibility
plt.switch_backend('Agg')

@dataclass
class OptimizedTurbulentConfig:
    L: float = 1.0
    R_cylinder: float = 0.5
    V_inf: float = 1.0
    cylinder_center: tuple[float, float] = (4.0, 2.0)
    x_min: float = 0.0
    x_max: float = 20.0
    y_min: float = 0.0
    y_max: float = 4.0
    nx: int = 480  # Increased for better vortex resolution
    ny: int = 144
    T_total: float = 30.0
    dt_base: float = 0.00025
    cfl_target: float = 0.15
    adaptive_dt: bool = True
    dt_min: float = 1e-6
    dt_max: float = 0.0005
    Re: float = 100.0
    use_les: bool = True
    smagorinsky_constant: float = 0.1  # Reduced to minimize damping
    use_supg: bool = True
    artificial_viscosity: float = 0.01  # Reduced to match physical viscosity
    pressure_iterations: int = 300
    pressure_tolerance: float = 1e-7
    max_velocity: float = 3.0
    initial_steps: int = 100
    parallel_threads: int = 4
    use_fast_pressure: bool = True
    memory_efficient: bool = True
    vectorized_ops: bool = True
    save_interval: int = 200
    output_dir: str = "v3_karman_vortex_results_2"
    dpi: int = 200

    def __post_init__(self):
        self.dx = (self.x_max - self.x_min) / (self.nx - 1)
        self.dy = (self.y_max - self.y_min) / (self.ny - 1)
        self.nu = 1.0 / self.Re
        self.dt = self.dt_base
        self.parallel_threads = min(self.parallel_threads, os.cpu_count())
        memory_mb = (self.nx * self.ny * 8 * 6) / (1024 * 1024)
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        print(f"--- Optimized Turbulent Flow Configuration ---")
        print(f"Reynolds Number: {self.Re}")
        print(f"Grid: {self.nx}x{self.ny} (optimized for vectorization)")
        print(f"Grid spacing: dx={self.dx:.4f}, dy={self.dy:.4f}")
        print(f"Parallel threads: {self.parallel_threads}")
        print(f"Estimated Memory: {memory_mb:.1f} MB")
        print(f"Available Memory: {available_mb:.1f} MB")
        print(f"Target performance: >15 steps/sec")
        print("--------------------------------------------\n")

@njit(parallel=True, fastmath=True, cache=True)
def compute_smagorinsky_viscosity_fast(u, v, dx, dy, cs):
    ny, nx = u.shape
    nu_t = np.zeros_like(u)
    delta = (dx * dy) ** 0.5
    cs_delta_sq = (cs * delta) ** 2
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            dudx = (u[i, j+1] - u[i, j]) / dx
            dudy = (u[i+1, j] - u[i, j]) / dy
            dvdx = (v[i, j+1] - v[i, j]) / dx
            dvdy = (v[i+1, j] - v[i, j]) / dy
            S_mag = (2 * (dudx*dudx + dvdy*dvdy) + (dudy + dvdx)**2) ** 0.5
            nu_t[i, j] = cs_delta_sq * S_mag
    return nu_t

@njit(parallel=True, fastmath=True, cache=True)
def compute_convection_fast(u, v, phi, dx, dy):
    ny, nx = phi.shape
    conv = np.zeros_like(phi)
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            u_vel = u[i, j]
            v_vel = v[i, j]
            dphidx = (phi[i, j] - phi[i, j-1]) * dx_inv if u_vel > 0 else (phi[i, j+1] - phi[i, j]) * dx_inv
            dphidy = (phi[i, j] - phi[i-1, j]) * dy_inv if v_vel > 0 else (phi[i+1, j] - phi[i, j]) * dy_inv
            conv[i, j] = u_vel * dphidx + v_vel * dphidy
    return conv

@njit(parallel=True, fastmath=True, cache=True)
def compute_convection_supg_fast(u, v, phi, dx, dy, tau_supg):
    ny, nx = phi.shape
    conv = np.zeros_like(phi)
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            u_vel = u[i, j]
            v_vel = v[i, j]
            dphidx = (phi[i, j+1] - phi[i, j-1]) * (0.5 * dx_inv)
            dphidy = (phi[i+1, j] - phi[i-1, j]) * (0.5 * dy_inv)
            conv_standard = u_vel * dphidx + v_vel * dphidy
            if tau_supg[i, j] > 0:
                d2phidx2 = (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) * (dx_inv * dx_inv)
                d2phidy2 = (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]) * (dy_inv * dy_inv)
                supg_term = tau_supg[i, j] * (u_vel * d2phidx2 + v_vel * d2phidy2)
                conv[i, j] = conv_standard - supg_term
            else:
                conv[i, j] = conv_standard
    return conv

@njit(parallel=True, fastmath=True, cache=True)
def compute_supg_stabilization_fast(u, v, dx, dy, dt, nu_eff):
    ny, nx = u.shape
    tau_supg = np.zeros_like(u)
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            vel_mag = (u[i, j]**2 + v[i, j]**2) ** 0.5
            h_elem = min(dx, dy)
            if vel_mag > 1e-10:
                Pe = vel_mag * h_elem / (nu_eff[i, j] + 1e-10)
                tau_supg[i, j] = h_elem / (2 * vel_mag) * min(1.0, Pe / 2.0)
            else:
                tau_supg[i, j] = dt / 2
    return tau_supg

@njit(parallel=True, fastmath=True, cache=True)
def compute_laplacian_fast(phi, dx, dy, nu_eff):
    ny, nx = phi.shape
    lap = np.zeros_like(phi)
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            lap[i, j] = nu_eff[i, j] * (
                (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) * dx2_inv +
                (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]) * dy2_inv
            )
    return lap

@njit(parallel=True, fastmath=True, cache=True)
def compute_divergence_fast(u, v, dx, dy):
    ny, nx = u.shape
    div = np.zeros_like(u)
    dx_inv = 0.5 / dx
    dy_inv = 0.5 / dy
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            div[i, j] = (u[i, j+1] - u[i, j-1]) * dx_inv + (v[i+1, j] - v[i-1, j]) * dy_inv
    return div

@njit(parallel=True, fastmath=True, cache=True)
def compute_gradient_fast(phi, dx, dy):
    ny, nx = phi.shape
    grad_x = np.zeros_like(phi)
    grad_y = np.zeros_like(phi)
    dx_inv = 0.5 / dx
    dy_inv = 0.5 / dy
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            grad_x[i, j] = (phi[i, j+1] - phi[i, j-1]) * dx_inv
            grad_y[i, j] = (phi[i+1, j] - phi[i-1, j]) * dy_inv
    return grad_x, grad_y

@njit(parallel=True, fastmath=True, cache=True)
def solve_pressure_gauss_seidel_fast(phi, div_u_star, dx, dy, dt, mask, iterations, tolerance):
    ny, nx = phi.shape
    dx2 = dx * dx
    dy2 = dy * dy
    dx2_inv = 1.0 / dx2
    dy2_inv = 1.0 / dy2
    denom_inv = 1.0 / (2.0 * (dx2_inv + dy2_inv))
    dt_inv = 1.0 / dt
    for iteration in range(iterations):
        max_change = 0.0
        for color in range(2):
            for i in prange(1, ny-1):
                for j in range(1 + (i + color) % 2, nx-1, 2):
                    if not mask[i, j]:
                        rhs = -div_u_star[i, j] * dt_inv
                        phi_new = (dx2_inv * (phi[i, j+1] + phi[i, j-1]) +
                                   dy2_inv * (phi[i+1, j] + phi[i-1, j]) - rhs) * denom_inv
                        change = abs(phi_new - phi[i, j])
                        if change > max_change:
                            max_change = change
                        phi[i, j] = phi_new
        if max_change < tolerance:
            break
    return phi

@njit(parallel=True, fastmath=True, cache=True)
def apply_ibm_fast(u, v, ibm_mask, force_strength):
    ny, nx = u.shape
    for i in prange(ny):
        for j in prange(nx):
            mask_val = ibm_mask[i, j]
            if mask_val > 0:
                u[i, j] *= (1.0 - mask_val * force_strength)
                v[i, j] *= (1.0 - mask_val * force_strength)
    return u, v

@njit(parallel=True, fastmath=True, cache=True)
def clean_divergence_fast(u, v, dx, dy, iterations=5):  # Reduced iterations
    ny, nx = u.shape
    phi = np.zeros_like(u)
    dx2 = dx * dx
    dy2 = dy * dy
    dx2_inv = 1.0 / dx2
    dy2_inv = 1.0 / dy2
    denom_inv = 1.0 / (2.0 * (dx2_inv + dy2_inv))
    for _ in range(iterations):
        div = compute_divergence_fast(u, v, dx, dy)
        for i in prange(1, ny-1):
            for j in prange(1, nx-1):
                phi[i, j] = (dx2_inv * (phi[i, j+1] + phi[i, j-1]) +
                            dy2_inv * (phi[i+1, j] + phi[i-1, j]) - div[i, j]) * denom_inv
        grad_x, grad_y = compute_gradient_fast(phi, dx, dy)
        u[1:-1, 1:-1] -= grad_x[1:-1, 1:-1]
        v[1:-1, 1:-1] -= grad_y[1:-1, 1:-1]
    return u, v

class OptimizedTurbulentSolver:
    def __init__(self, config: OptimizedTurbulentConfig):
        self.config = config
        self.setup_grid()
        self.setup_boundary_masks()
        self.initialize_fields()
        self.step = 0
        
    def setup_grid(self):
        cfg = self.config
        self.x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
        self.y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        
    def setup_boundary_masks(self):
        cfg = self.config
        x_c, y_c = cfg.cylinder_center
        self.dist = np.sqrt((self.X - x_c)**2 + (self.Y - y_c)**2)
        self.cylinder_mask = self.dist <= cfg.R_cylinder
        sigma = 2 * cfg.dx
        self.ibm_mask = np.exp(-((self.dist - cfg.R_cylinder) / sigma)**2)
        self.ibm_mask = np.where(self.dist < cfg.R_cylinder, 1.0, 
                                np.where(self.dist < cfg.R_cylinder + 5*cfg.dx, self.ibm_mask, 0.0))
        
    def initialize_fields(self):
        cfg = self.config
        dtype = np.float32 if cfg.memory_efficient else np.float64
        self.u = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.v = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.p = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.nu_t = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.tau_supg = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)
        self.div_u_star = np.zeros_like(self.u)
        self.phi = np.zeros_like(self.u)
        self.initialize_potential_flow()
        
    def initialize_potential_flow(self):
        cfg = self.config
        x_c, y_c = cfg.cylinder_center
        for i in range(cfg.ny):
            for j in range(cfg.nx):
                r = self.dist[i, j]
                mask_val = self.ibm_mask[i, j]
                if r > cfg.R_cylinder + 4*cfg.dx:
                    theta = np.arctan2(self.Y[i, j] - y_c, self.X[i, j] - x_c)
                    factor = (cfg.R_cylinder / r)**2
                    self.u[i, j] = cfg.V_inf * (1 - factor * np.cos(2 * theta)) * (1 - mask_val)
                    self.v[i, j] = -cfg.V_inf * factor * np.sin(2 * theta) * (1 - mask_val)
                else:
                    blend = min(1.0, ((r - cfg.R_cylinder) / (4 * cfg.dx))**2)
                    self.u[i, j] = cfg.V_inf * blend * (1 - mask_val)
                    self.v[i, j] = 0.0
        
    def adaptive_time_step(self):
        if not self.config.adaptive_dt:
            return self.config.dt_base
        cfg = self.config
        vel_max = max(np.max(np.abs(self.u)), np.max(np.abs(self.v)), 1e-10)
        dt_cfl = cfg.cfl_target * min(cfg.dx, cfg.dy) / vel_max
        nu_total = cfg.nu + np.mean(self.nu_t) + cfg.artificial_viscosity
        dt_visc = 0.4 * min(cfg.dx, cfg.dy)**2 / nu_total
        return np.clip(min(dt_cfl, dt_visc), cfg.dt_min, cfg.dt_max)
        
    def solve_pressure_fast(self, div_u_star):
        cfg = self.config
        if cfg.use_fast_pressure:
            self.phi.fill(0.0)
            self.phi = solve_pressure_gauss_seidel_fast(
                self.phi, div_u_star, cfg.dx, cfg.dy, cfg.dt,
                self.cylinder_mask, cfg.pressure_iterations, cfg.pressure_tolerance
            )
        else:
            self.phi.fill(0.0)
            for _ in range(cfg.pressure_iterations):
                phi_new = self.phi.copy()
                phi_new[1:-1, 1:-1] = 0.25 * (
                    self.phi[1:-1, 2:] + self.phi[1:-1, :-2] +
                    self.phi[2:, 1:-1] + self.phi[:-2, 1:-1] -
                    cfg.dx**2 * div_u_star[1:-1, 1:-1] / cfg.dt
                )
                phi_new[self.cylinder_mask] = 0
                self.phi = phi_new
        return self.phi
        
    def apply_boundary_conditions(self, u, v):
        cfg = self.config
        pert_scale = min(1.0, (self.step - cfg.initial_steps) / cfg.initial_steps) if self.step > cfg.initial_steps else 0.0
        perturbation = 0.005 * pert_scale * np.sin(2 * np.pi * self.y / cfg.y_max + 0.02 * self.step)
        u[:, 0] = cfg.V_inf * (1 + perturbation)
        v[:, 0] = 0
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0
        
    def time_step(self):
        cfg = self.config
        dt = self.adaptive_time_step()
        u_old = self.u.copy()
        v_old = self.v.copy()
        
        if cfg.use_les:
            self.nu_t = compute_smagorinsky_viscosity_fast(
                u_old, v_old, cfg.dx, cfg.dy, cfg.smagorinsky_constant
            )
        
        nu_eff = cfg.nu + self.nu_t + cfg.artificial_viscosity
        if cfg.use_supg:
            self.tau_supg = compute_supg_stabilization_fast(
                u_old, v_old, cfg.dx, cfg.dy, dt, nu_eff
            )
            conv_u = compute_convection_supg_fast(u_old, v_old, u_old, cfg.dx, cfg.dy, self.tau_supg)
            conv_v = compute_convection_supg_fast(u_old, v_old, v_old, cfg.dx, cfg.dy, self.tau_supg)
        else:
            conv_u = compute_convection_fast(u_old, v_old, u_old, cfg.dx, cfg.dy)
            conv_v = compute_convection_fast(u_old, v_old, v_old, cfg.dx, cfg.dy)
        
        lap_u = compute_laplacian_fast(u_old, cfg.dx, cfg.dy, nu_eff)
        lap_v = compute_laplacian_fast(v_old, cfg.dx, cfg.dy, nu_eff)
        
        self.u_star[:] = u_old + dt * (-conv_u + lap_u)
        self.v_star[:] = v_old + dt * (-conv_v + lap_v)
        
        self.apply_boundary_conditions(self.u_star, self.v_star)
        force_strength = min(1.0, self.step / cfg.initial_steps)
        self.u_star, self.v_star = apply_ibm_fast(self.u_star, self.v_star, self.ibm_mask, force_strength)
        
        self.div_u_star = compute_divergence_fast(self.u_star, self.v_star, cfg.dx, cfg.dy)
        print(f"Step {self.step}: Pre-pressure divergence = {np.max(np.abs(self.div_u_star)):.3f}")
        self.phi = self.solve_pressure_fast(self.div_u_star)
        
        dpdx, dpdy = compute_gradient_fast(self.phi, cfg.dx, cfg.dy)
        grad_mag = np.sqrt(dpdx**2 + dpdy**2)
        print(f"Step {self.step}: Max pressure gradient = {np.max(np.abs(grad_mag)):.3f}")
        self.u[:] = self.u_star - dt * dpdx
        self.v[:] = self.v_star - dt * dpdy
        
        self.u, self.v = clean_divergence_fast(self.u, self.v, cfg.dx, cfg.dy, iterations=5)
        
        post_div = compute_divergence_fast(self.u, self.v, cfg.dx, cfg.dy)
        print(f"Step {self.step}: Post-pressure divergence = {np.max(np.abs(post_div)):.3f}")
        
        self.apply_boundary_conditions(self.u, self.v)
        self.u, self.v = apply_ibm_fast(self.u, self.v, self.ibm_mask, force_strength)
        
        vorticity = self.compute_vorticity()
        vort_max = np.nanmax(np.abs(vorticity))
        print(f"Step {self.step}: Max vorticity = {vort_max:.3f}")
        
        np.clip(self.u, -cfg.max_velocity, cfg.max_velocity, out=self.u)
        np.clip(self.v, -cfg.max_velocity, cfg.max_velocity, out=self.v)
        
        self.step += 1
        
    def compute_vorticity(self):
        cfg = self.config
        vorticity = np.zeros_like(self.u)
        vorticity[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * cfg.dx) -
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * cfg.dy)
        )
        vorticity[self.cylinder_mask] = np.nan
        return vorticity

class OptimizedVisualizer:
    def __init__(self, config: OptimizedTurbulentConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=2)
        plt.style.use('seaborn-v0_8')
        
    def plot_velocity_frame(self, solver: OptimizedTurbulentSolver, step: int, current_time: float):
        cfg = self.config
        try:
            fig = plt.figure(figsize=(12, 6), facecolor='white')
            ax = fig.add_subplot(111)
            vel_mag = np.sqrt(solver.u**2 + solver.v**2)
            levels = np.linspace(0, np.nanmax(vel_mag)*0.9, 31)
            cf = ax.contourf(solver.X, solver.Y, vel_mag, levels=levels, cmap='viridis')
            plt.colorbar(cf, ax=ax, label='Dimensionless Velocity Magnitude |V|')
            try:
                skip = max(6, min(solver.X.shape) // 30)  # Denser streamlines
                seed_points = np.array([
                    [cfg.x_min + 1, y] for y in np.linspace(cfg.y_min + 0.3, cfg.y_max - 0.3, 16)
                ])
                ax.streamplot(solver.X, solver.Y, solver.u, solver.v, 
                              color='white', linewidth=0.8, density=2.0,
                              start_points=seed_points, maxlength=50)
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
                gc.collect()
        except Exception as e:
            print(f"Error plotting velocity frame {step}: {e}")
            plt.close('all')
    
    def plot_vorticity_frame(self, solver: OptimizedTurbulentSolver, step: int, current_time: float):
        cfg = self.config
        try:
            fig = plt.figure(figsize=(12, 6), facecolor='white')
            ax = fig.add_subplot(111)
            vorticity = solver.compute_vorticity()
            vort_max = np.nanmax(np.abs(vorticity))
            levels = np.linspace(-vort_max*0.9, vort_max*0.9, 51)  # Adjusted for sensitivity
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
                gc.collect()
        except Exception as e:
            print(f"Error plotting vorticity frame {step}: {e}")
            plt.close('all')
            
    def cleanup(self):
        self.executor.shutdown(wait=False)

def monitor_simulation_health(solver, step):
    cfg = solver.config
    if np.any(~np.isfinite(solver.u)) or np.any(~np.isfinite(solver.v)):
        print(f"ERROR: Non-finite values at step {step}")
        return False
    vel_max = max(np.max(np.abs(solver.u)), np.max(np.abs(solver.v)))
    if vel_max > cfg.max_velocity:
        print(f"WARNING: High velocity {vel_max:.3f} at step {step}")
        return False
    div_max = np.max(np.abs(compute_divergence_fast(solver.u, solver.v, cfg.dx, cfg.dy)))
    div_threshold = 20.0 if step <= cfg.initial_steps else 5.0
    if div_max > div_threshold:
        print(f"WARNING: High divergence {div_max:.3f} at step {step}")
        return False
    return True

def main():
    config = OptimizedTurbulentConfig(
        Re=100.0,
        nx=480,
        ny=144,
        T_total=30.0,
        use_les=True,
        use_supg=True,
        use_fast_pressure=True,
        adaptive_dt=True,
        memory_efficient=True
    )
    
    print("🚀 Initializing Ultra-High-Performance Turbulent CFD Solver...")
    print(f"Target: >15 steps/second with full turbulence modeling\n")
    
    solver = OptimizedTurbulentSolver(config)
    visualizer = OptimizedVisualizer(config)
    
    start_time = time.time()
    step = 0
    current_time = 0.0
    
    try:
        print("🏁 Starting optimized simulation...")
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
        print("\n⏹️  Simulation interrupted by user.")
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        final_speed = step / total_time if total_time > 0 else 0
        
        print(f"\n🏆 Simulation Performance Report:")
        print(f"  Total steps: {step}")
        print(f"  Final time: {current_time:.2f}")
        print(f"  Wall time: {total_time/60:.1f} minutes")
        print(f"  Average speed: {final_speed:.1f} steps/second")
        print(f"  Results saved in: {visualizer.output_path}")
        
        visualizer.cleanup()
        del solver, visualizer
        gc.collect()

if __name__ == "__main__":
    main()