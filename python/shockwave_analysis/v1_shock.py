# MIT License
# Copyright (c) 2025 Santhosh S
# See LICENSE file for full license text.

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import os
import time
from pathlib import Path
from dataclasses import dataclass
import h5py
import logging
from tqdm import tqdm
import gc

# Set matplotlib backend
plt.switch_backend('Agg')
plt.style.use('dark_background')

# Configure logging (minimal terminal output)
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/v1_ma_2.log')]
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

@dataclass
class ShockwaveConfig:
    L: float = 1.0
    x_min: float = 0.0
    x_max: float = 2.0
    y_min: float = 0.0
    y_max: float = 1.0
    nx: int = 400
    ny: int = 200
    M_inf: float = 2.0
    p_inf: float = 1.0
    rho_inf: float = 1.0
    gamma: float = 1.4
    T_total: float = 5.0
    cfl: float = 0.99
    wedge_angle: float = np.deg2rad(10.0)
    wedge_start_x: float = 0.5
    save_interval: int = 200
    output_dir: str = "v1_ma_2"
    hdf5_file: str = f"{output_dir}.h5"
    dpi: int = 200
    memory_efficient: bool = True
    parallel_threads: int = 4
    epsilon: float = 1e-8
    max_val: float = 1e3
    solver_type: str = "hllc"  # Options: "roe", "hllc"

    def __post_init__(self):
        self.dx = (self.x_max - self.x_min) / self.nx
        self.dy = (self.y_max - self.y_min) / self.ny
        self.parallel_threads = min(self.parallel_threads, os.cpu_count())
        self.a_inf = np.sqrt(self.gamma * self.p_inf / self.rho_inf)
        self.u_inf = self.M_inf * self.a_inf
        logger.info(f"Grid: {self.nx}x{self.ny}, Mach: {self.M_inf}, Solver: {self.solver_type}")

@njit(fastmath=True, cache=True)
def scalar_clip(x, a, b):
    return max(a, min(x, b))

@njit(fastmath=True, cache=True)
def array_clip(x, a, b):
    return np.minimum(np.maximum(x, a), b)

@njit(fastmath=True, cache=True)
def superbee(a, b):
    if a * b <= 0:
        return 0.0
    r = a / (b + 1e-10)
    return max(0.0, min(2 * r, 1.0), min(r, 2.0)) * b

@njit(fastmath=True, cache=True)
def compute_fluxes(U, gamma, epsilon, max_val):
    rho = max(U[0], epsilon)
    u = scalar_clip(U[1] / rho, -max_val, max_val)
    v = scalar_clip(U[2] / rho, -max_val, max_val)
    E = scalar_clip(U[3] / rho, epsilon, max_val)
    p = max((gamma - 1) * rho * (E - 0.5 * (u**2 + v**2)), epsilon)
    
    F = np.array([rho * u, rho * u * u + p, rho * u * v, rho * u * (E + p / rho)], dtype=np.float32)
    G = np.array([rho * v, rho * v * u, rho * v * v + p, rho * v * (E + p / rho)], dtype=np.float32)
    
    return F, G

@njit(fastmath=True, cache=True)
def roe_solver(U_L, U_R, gamma, normal_x, normal_y, epsilon, max_val):
    U_L = U_L.astype(np.float32)
    U_R = U_R.astype(np.float32)
    rho_L = max(U_L[0], epsilon)
    if rho_L <= epsilon:
        return np.zeros(4, dtype=np.float32)
    u_L = scalar_clip(U_L[1] / rho_L, -max_val, max_val)
    v_L = scalar_clip(U_L[2] / rho_L, -max_val, max_val)
    E_L = scalar_clip(U_L[3] / rho_L, epsilon, max_val)
    p_L = max((gamma - 1) * rho_L * (E_L - 0.5 * (u_L**2 + v_L**2)), epsilon)
    h_L = (U_L[3] + p_L) / rho_L
    
    rho_R = max(U_R[0], epsilon)
    if rho_R <= epsilon:
        return np.zeros(4, dtype=np.float32)
    u_R = scalar_clip(U_R[1] / rho_R, -max_val, max_val)
    v_R = scalar_clip(U_R[2] / rho_R, -max_val, max_val)
    E_R = scalar_clip(U_R[3] / rho_R, epsilon, max_val)
    p_R = max((gamma - 1) * rho_R * (E_R - 0.5 * (u_R**2 + v_R**2)), epsilon)
    h_R = (U_R[3] + p_R) / rho_R
    
    delta_rho = superbee(rho_R - rho_L, rho_L - max(U_L[0], epsilon))
    delta_u = superbee(u_R - u_L, u_L - U_L[1]/max(U_L[0], epsilon))
    delta_v = superbee(v_R - v_L, v_L - U_L[2]/max(U_L[0], epsilon))
    
    rho_roe = np.sqrt(rho_L * rho_R)
    u_roe = (u_L * np.sqrt(rho_L) + u_R * np.sqrt(rho_R)) / (np.sqrt(rho_L) + np.sqrt(rho_R) + epsilon)
    v_roe = (v_L * np.sqrt(rho_L) + v_R * np.sqrt(rho_R)) / (np.sqrt(rho_L) + np.sqrt(rho_R) + epsilon)
    h_roe = (h_L * np.sqrt(rho_L) + h_R * np.sqrt(rho_R)) / (np.sqrt(rho_L) + np.sqrt(rho_R) + epsilon)
    a_roe = np.sqrt(max((gamma - 1) * (h_roe - 0.5 * (u_roe**2 + v_roe**2)), epsilon))
    
    vel_n = u_roe * normal_x + v_roe * normal_y
    delta_U = (U_R - U_L).astype(np.float32)
    
    lambda1 = abs(vel_n)
    lambda2 = abs(vel_n + a_roe)
    lambda3 = abs(vel_n - a_roe)
    epsilon_roe = 0.05 * a_roe
    lambda1 = max(epsilon_roe, lambda1)
    lambda2 = max(epsilon_roe, lambda2)
    lambda3 = max(epsilon_roe, lambda3)
    
    F_L, _ = compute_fluxes(U_L, gamma, epsilon, max_val)
    F_R, _ = compute_fluxes(U_R, gamma, epsilon, max_val)
    F_avg = 0.5 * (F_L + F_R)
    
    flux = (F_avg - 0.5 * (lambda1 + lambda2 + lambda3) * delta_U).astype(np.float32)
    return flux

@njit(fastmath=True, cache=True)
def hllc_solver(U_L, U_R, gamma, normal_x, normal_y, epsilon, max_val):
    U_L = U_L.astype(np.float32)
    U_R = U_R.astype(np.float32)
    rho_L = max(U_L[0], epsilon)
    if rho_L <= epsilon:
        return np.zeros(4, dtype=np.float32)
    u_L = scalar_clip(U_L[1] / rho_L, -max_val, max_val)
    v_L = scalar_clip(U_L[2] / rho_L, -max_val, max_val)
    E_L = scalar_clip(U_L[3] / rho_L, epsilon, max_val)
    p_L = max((gamma - 1) * rho_L * (E_L - 0.5 * (u_L**2 + v_L**2)), epsilon)
    
    rho_R = max(U_R[0], epsilon)
    if rho_R <= epsilon:
        return np.zeros(4, dtype=np.float32)
    u_R = scalar_clip(U_R[1] / rho_R, -max_val, max_val)
    v_R = scalar_clip(U_R[2] / rho_R, -max_val, max_val)
    E_R = scalar_clip(U_R[3] / rho_R, epsilon, max_val)
    p_R = max((gamma - 1) * rho_R * (E_R - 0.5 * (u_R**2 + v_R**2)), epsilon)
    
    q_L = u_L * normal_x + v_L * normal_y
    q_R = u_R * normal_x + v_R * normal_y
    a_L = np.sqrt(max(gamma * p_L / rho_L, epsilon))
    a_R = np.sqrt(max(gamma * p_R / rho_R, epsilon))
    
    S_L = min(q_L - a_L, q_R - a_R)
    S_R = max(q_L + a_L, q_R + a_R)
    S_M = (rho_R * q_R * (S_R - q_R) - rho_L * q_L * (S_L - q_L) + p_L - p_R) / (rho_R * (S_R - q_R) - rho_L * (S_L - q_L) + epsilon)
    
    F_L, G_L = compute_fluxes(U_L, gamma, epsilon, max_val)
    F_R, G_R = compute_fluxes(U_R, gamma, epsilon, max_val)
    
    if normal_x == 1.0:
        flux_L = F_L
        flux_R = F_R
    else:
        flux_L = G_L
        flux_R = G_R
    
    if S_L >= 0:
        return flux_L
    if S_R <= 0:
        return flux_R
    
    p_star = rho_L * (q_L - S_L) * (q_L - S_M) + p_L
    U_star_L = np.array([
        rho_L * (S_L - q_L) / (S_L - S_M + epsilon),
        rho_L * (S_L - q_L) / (S_L - S_M + epsilon) * (S_M * normal_x),
        rho_L * (S_L - q_L) / (S_L - S_M + epsilon) * (v_L if normal_x == 1.0 else S_M),
        rho_L * (S_L - q_L) / (S_L - S_M + epsilon) * (E_L + (p_star * S_M - p_L * q_L) / (rho_L * (S_L - q_L) + epsilon))
    ], dtype=np.float32)
    
    if S_M >= 0:
        return (flux_L + S_L * (U_star_L - U_L)).astype(np.float32)
    
    U_star_R = np.array([
        rho_R * (S_R - q_R) / (S_R - S_M + epsilon),
        rho_R * (S_R - q_R) / (S_R - S_M + epsilon) * (S_M * normal_x),
        rho_R * (S_R - q_R) / (S_R - S_M + epsilon) * (v_R if normal_x == 1.0 else S_M),
        rho_R * (S_R - q_R) / (S_R - S_M + epsilon) * (E_R + (p_star * S_M - p_R * q_R) / (rho_R * (S_R - q_R) + epsilon))
    ], dtype=np.float32)
    
    return (flux_R + S_R * (U_star_R - U_R)).astype(np.float32)

@njit(parallel=True, fastmath=True, cache=True)
def update_state(U, F, G, dx, dy, dt, epsilon, max_val):
    nx, ny, _ = U.shape
    U_new = np.zeros_like(U, dtype=np.float32)
    for i in prange(1, nx-1):
        for j in prange(1, ny-1):
            update = U[i, j] - dt/dx * (F[i, j] - F[i-1, j]) - dt/dy * (G[i, j] - G[i, j-1])
            update[0] = max(update[0], epsilon)
            update[1] = scalar_clip(update[1], -max_val * update[0], max_val * update[0])
            update[2] = scalar_clip(update[2], -max_val * update[0], max_val * update[0])
            update[3] = max(update[3], epsilon * update[0])
            U_new[i, j] = update
    return U_new

class ShockwaveSolver:
    def __init__(self, config: ShockwaveConfig):
        self.config = config
        self.setup_grid()
        self.setup_wedge()
        self.initialize_fields()
        self.step = 0
        self.times = []

    def setup_grid(self):
        cfg = self.config
        self.x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
        self.y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def setup_wedge(self):
        cfg = self.config
        self.wedge_mask = np.zeros((cfg.nx, cfg.ny), dtype=np.bool_)
        wedge_y = lambda x: np.tan(cfg.wedge_angle) * (x - cfg.wedge_start_x)
        for i in range(cfg.nx):
            for j in range(cfg.ny):
                x, y = self.X[i, j], self.Y[i, j]
                if x >= cfg.wedge_start_x and y <= wedge_y(x):
                    self.wedge_mask[i, j] = True

    def initialize_fields(self):
        cfg = self.config
        self.U = np.zeros((cfg.nx, cfg.ny, 4), dtype=np.float32)
        u = cfg.u_inf
        v = 0.0
        p = cfg.p_inf
        rho = cfg.rho_inf
        E = p / (rho * (cfg.gamma - 1)) + 0.5 * (u**2 + v**2)
        self.U[:, :, 0] = rho
        self.U[:, :, 1] = rho * u
        self.U[:, :, 2] = rho * v
        self.U[:, :, 3] = rho * E

    def compute_dt(self):
        cfg = self.config
        rho = np.maximum(self.U[:, :, 0], cfg.epsilon)
        u = array_clip(self.U[:, :, 1] / rho, -cfg.max_val, cfg.max_val)
        v = array_clip(self.U[:, :, 2] / rho, -cfg.max_val, cfg.max_val)
        E = array_clip(self.U[:, :, 3] / rho, cfg.epsilon, cfg.max_val)
        p = np.maximum((cfg.gamma - 1) * rho * (E - 0.5 * (u**2 + v**2)), cfg.epsilon)
        a = np.sqrt(array_clip(cfg.gamma * p / rho, cfg.epsilon, cfg.max_val))
        max_speed_x = np.minimum(np.max(np.abs(u) + a), cfg.max_val)
        max_speed_y = np.minimum(np.max(np.abs(v) + a), cfg.max_val)
        dt_x = cfg.cfl * cfg.dx / max(max_speed_x, cfg.epsilon)
        dt_y = cfg.cfl * cfg.dy / max(max_speed_y, cfg.epsilon)
        return min(dt_x, dt_y)

    def apply_boundary_conditions(self):
        cfg = self.config
        self.U[0, :, 0] = cfg.rho_inf
        self.U[0, :, 1] = cfg.rho_inf * cfg.u_inf
        self.U[0, :, 2] = 0.0
        self.U[0, :, 3] = cfg.rho_inf * (cfg.p_inf / (cfg.rho_inf * (cfg.gamma - 1)) + 0.5 * cfg.u_inf**2)
        self.U[-1, :, :] = self.U[-2, :, :]
        for j in range(cfg.ny):
            if self.Y[0, j] <= np.tan(cfg.wedge_angle) * (self.X[0, j] - cfg.wedge_start_x):
                self.U[:, j, 2] = -self.U[:, j, 2]
            else:
                self.U[:, 0, :] = self.U[:, 1, :]
                self.U[:, -1, :] = self.U[:, -2, :]

    def time_step(self):
        cfg = self.config
        dt = self.compute_dt()
        if not np.isfinite(dt):
            logger.warning(f"Invalid dt at step {self.step}")
            return 0.0
        
        F = np.zeros_like(self.U, dtype=np.float32)
        G = np.zeros_like(self.U, dtype=np.float32)
        solver_func = hllc_solver if cfg.solver_type == "hllc" else roe_solver
        for i in prange(1, cfg.nx):
            for j in prange(cfg.ny):
                F[i-1, j] = solver_func(self.U[i-1, j], self.U[i, j], cfg.gamma, 1.0, 0.0, cfg.epsilon, cfg.max_val)
        for i in prange(cfg.nx):
            for j in prange(1, cfg.ny):
                G[i, j-1] = solver_func(self.U[i, j-1], self.U[i, j], cfg.gamma, 0.0, 1.0, cfg.epsilon, cfg.max_val)
        
        self.U = update_state(self.U, F, G, cfg.dx, cfg.dy, dt, cfg.epsilon, cfg.max_val)
        
        self.apply_boundary_conditions()
        
        self.U[self.wedge_mask, 1] = 0.0
        self.U[self.wedge_mask, 2] = 0.0
        
        self.times.append(self.step * dt)
        self.step += 1
        return dt

    def check_health(self):
        cfg = self.config
        if np.any(~np.isfinite(self.U)):
            logger.warning(f"Non-finite values at step {self.step}")
            return False
        rho = self.U[:, :, 0]
        if np.any(rho < 0):
            logger.warning(f"Negative density at step {self.step}")
            return False
        return True

class ShockwaveVisualizer:
    def __init__(self, config: ShockwaveConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        self.density_path = self.output_path / "density_frames"
        self.mach_path = self.output_path / "mach_frames"
        self.vorticity_path = self.output_path / "vorticity_frames"
        self.density_path.mkdir(exist_ok=True)
        self.mach_path.mkdir(exist_ok=True)
        self.vorticity_path.mkdir(exist_ok=True)

    def save_data_to_hdf5(self, solver: ShockwaveSolver, step: int, current_time: float):
        cfg = self.config
        try:
            with h5py.File(self.output_path / cfg.hdf5_file, 'a') as f:
                group_name = f"step_{step:06d}"
                if group_name not in f:
                    group = f.create_group(group_name)
                    group.attrs['time'] = current_time
                    group.create_dataset('U', data=solver.U, compression='gzip', compression_opts=4)
                    group.create_dataset('X', data=solver.X, compression='gzip', compression_opts=4)
                    group.create_dataset('Y', data=solver.Y, compression='gzip', compression_opts=4)
        except Exception as e:
            logger.warning(f"Error saving HDF5 data: {e}")

    def generate_frames_from_hdf5(self):
        cfg = self.config
        output_path = self.output_path
        hdf5_path = output_path / cfg.hdf5_file
        if not hdf5_path.exists():
            logger.warning(f"HDF5 file {hdf5_path} not found")
            return
        with h5py.File(hdf5_path, 'r') as f:
            steps = sorted([k for k in f.keys() if k.startswith('step_')],
                           key=lambda x: int(x.split('_')[1]))
            for step_key in tqdm(steps, desc="Generating Frames", unit="frame"):
                step = int(step_key.split('_')[1])
                group = f[step_key]
                U = group['U'][:]
                X = group['X'][:]
                Y = group['Y'][:]
                current_time = group.attrs['time']
                
                rho = np.maximum(U[:, :, 0], cfg.epsilon)
                u = np.clip(U[:, :, 1] / rho, -cfg.max_val, cfg.max_val)
                v = np.clip(U[:, :, 2] / rho, -cfg.max_val, cfg.max_val)
                E = np.clip(U[:, :, 3] / rho, cfg.epsilon, cfg.max_val)
                p = np.maximum((cfg.gamma - 1) * rho * (E - 0.5 * (u**2 + v**2)), cfg.epsilon)
                a = np.sqrt(np.clip(cfg.gamma * p / rho, cfg.epsilon, cfg.max_val))
                mach = np.sqrt(u**2 + v**2) / a
                
                vorticity = np.zeros_like(rho)
                dx, dy = cfg.dx, cfg.dy
                for i in range(1, cfg.nx-1):
                    for j in range(1, cfg.ny-1):
                        vorticity[i, j] = (v[i+1, j] - v[i-1, j]) / (2 * dx) - (u[i, j+1] - u[i, j-1]) / (2 * dy)
                
                try:
                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot(111)
                    levels = np.linspace(np.min(rho), np.max(rho), 31)
                    cf = ax.contourf(X, Y, rho, levels=levels, cmap='viridis')
                    plt.colorbar(cf, ax=ax, label='Density', shrink=0.8)
                    ax.plot([cfg.wedge_start_x, cfg.x_max], [0, np.tan(cfg.wedge_angle) * (cfg.x_max - cfg.wedge_start_x)], 'w-', lw=2)
                    ax.set_xlim(cfg.x_min, cfg.x_max)
                    ax.set_ylim(cfg.y_min, cfg.y_max)
                    ax.set_aspect('equal')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_title(f'Density, M={cfg.M_inf:.1f}, t={current_time:.2f}')
                    plt.tight_layout()
                    filename = self.density_path / f"density_frame_{step:06d}.png"
                    plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error plotting density frame: {e}")
                    plt.close('all')
                
                try:
                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot(111)
                    levels = np.linspace(0, np.max(mach)*0.9, 31)
                    cf = ax.contourf(X, Y, mach, levels=levels, cmap='magma')
                    plt.colorbar(cf, ax=ax, label='Mach Number', shrink=0.8)
                    ax.plot([cfg.wedge_start_x, cfg.x_max], [0, np.tan(cfg.wedge_angle) * (cfg.x_max - cfg.wedge_start_x)], 'w-', lw=2)
                    ax.set_xlim(cfg.x_min, cfg.x_max)
                    ax.set_ylim(cfg.y_min, cfg.y_max)
                    ax.set_aspect('equal')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_title(f'Mach Number, M={cfg.M_inf:.1f}, t={current_time:.2f}')
                    plt.tight_layout()
                    filename = self.mach_path / f"mach_frame_{step:06d}.png"
                    plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error plotting Mach frame: {e}")
                    plt.close('all')
                
                try:
                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot(111)
                    levels = np.linspace(np.min(vorticity), np.max(vorticity), 31)
                    cf = ax.contourf(X, Y, vorticity, levels=levels, cmap='inferno')
                    plt.colorbar(cf, ax=ax, label='Vorticity', shrink=0.8)
                    ax.plot([cfg.wedge_start_x, cfg.x_max], [0, np.tan(cfg.wedge_angle) * (cfg.x_max - cfg.wedge_start_x)], 'w-', lw=2)
                    ax.set_xlim(cfg.x_min, cfg.x_max)
                    ax.set_ylim(cfg.y_min, cfg.y_max)
                    ax.set_aspect('equal')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_title(f'Vorticity, M={cfg.M_inf:.1f}, t={current_time:.2f}')
                    plt.tight_layout()
                    filename = self.vorticity_path / f"vorticity_frame_{step:06d}.png"
                    plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error plotting vorticity frame: {e}")
                    plt.close('all')
                
                if cfg.memory_efficient:
                    gc.collect()

def main():
    config = ShockwaveConfig(
        M_inf=2.0,
        nx=400,
        ny=200,
        T_total=5.0,
        cfl=0.99,
        wedge_angle=np.deg2rad(10.0),
        save_interval=200,
        memory_efficient=True,
        solver_type="hllc"
    )

    solver = ShockwaveSolver(config)
    visualizer = ShockwaveVisualizer(config)

    start_time = time.time()
    step = 0
    current_time = 0.0

    try:
        visualizer.save_data_to_hdf5(solver, step, current_time)
        with tqdm(total=config.T_total, desc="Simulation Progress", unit="time",
                  bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]") as pbar:
            while current_time < config.T_total:
                if step % 100 == 0:
                    if not solver.check_health():
                        logger.warning("Simulation unstable. Stopping.")
                        break
                dt = solver.time_step()
                if dt == 0.0 or not np.isfinite(dt):
                    logger.warning(f"Invalid dt at step {step}. Stopping.")
                    break
                current_time += dt
                if step % config.save_interval == 0:
                    visualizer.save_data_to_hdf5(solver, step, current_time)
                step += 1
                pbar.update(float(dt))

        visualizer.generate_frames_from_hdf5()

    except Exception as e:
        logger.warning(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Steps: {step}, Time: {current_time:.2f}, Wall: {total_time/60:.1f} min")

if __name__ == "__main__":
    main()