import numpy as np
import numba
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
from dataclasses import dataclass
import psutil
import logging
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc
import warnings

# Configure matplotlib and logging
plt.switch_backend('Agg')
plt.style.use('dark_background')
warnings.filterwarnings('ignore')

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/cavity_flow_v1.log')]
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console)

@dataclass
class CavityFlowConfig:
    NX: int = 600
    NY: int = 180
    NG: int = 2
    DOMAIN_WIDTH: float = 2.0
    DOMAIN_HEIGHT: float = 1.0
    CAVITY_LENGTH: float = 0.5
    L_D: float = 2.0
    MACH_INF: float = 2.5
    P_INF: float = 1.0
    RHO_INF: float = 1.0
    GAMMA: float = 1.4
    T_FINAL: float = 10.0
    CFL: float = 0.3
    CFL_MIN: float = 0.05
    CFL_MAX: float = 0.5
    SAVE_INTERVAL: int = 200
    OUTPUT_DIR: str = "cavity_flow_v1"
    HDF5_FILE: str = f"{OUTPUT_DIR}.h5"
    DPI: int = 200
    EPSILON: float = 1e-8
    P_MIN: float = 1e-8
    RHO_MIN: float = 1e-8
    MAX_VAL: float = 100.0
    PARALLEL_THREADS: int = min(16, os.cpu_count())
    MEMORY_EFFICIENT: bool = True
    USE_FAST_SOLVER: bool = True
    ARTIFICIAL_VISCOSITY: float = 0.001
    PRESSURE_ITERATIONS: int = 1500
    PRESSURE_TOLERANCE: float = 1e-8

    def __post_init__(self):
        self.CAVITY_DEPTH = self.CAVITY_LENGTH / self.L_D
        self.dx = self.DOMAIN_WIDTH / (self.NX - 1)
        self.dy = self.DOMAIN_HEIGHT / (self.NY - 1)
        self.a_inf = np.sqrt(self.GAMMA * self.P_INF / self.RHO_INF)
        self.u_inf = self.MACH_INF * self.a_inf
        memory_mb = (self.NX * self.NY * 4 * 8) / (1024 * 1024)
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        logger.info("--- Optimized Supersonic Cavity Flow Configuration ---")
        logger.info(f"Mach Number: {self.MACH_INF}")
        logger.info(f"Grid: {self.NX}x{self.NY}")
        logger.info(f"Cavity L/D: {self.L_D}")
        logger.info(f"Grid spacing: dx={self.dx:.4f}, dy={self.dy:.4f}")
        logger.info(f"Parallel threads: {self.PARALLEL_THREADS}")
        logger.info(f"Estimated Memory: {memory_mb:.1f} MB")
        logger.info(f"Available Memory: {available_mb:.1f} MB")
        logger.info(f"Target performance: >100 steps/sec")
        logger.info("-" * 55)

@jit(nopython=True, cache=True, fastmath=True)
def minmod(a, b):
    cond1 = np.logical_and(np.abs(a) < np.abs(b), a * b > 0)
    cond2 = np.logical_and(np.abs(b) < np.abs(a), a * b > 0)
    return np.where(cond1, a, np.where(cond2, b, 0.0))

@jit(nopython=True, cache=True, fastmath=True)
def cons_to_prim_limited(U, EPSILON, P_MIN, RHO_MIN, GAMMA, MAX_VAL, start_i, end_i, start_j, end_j):
    ny, nx, _ = U.shape
    rho = np.maximum(U[start_i:end_i, start_j:end_j, 0], RHO_MIN)
    inv_rho = 1.0 / (rho + EPSILON)
    u = U[start_i:end_i, start_j:end_j, 1] * inv_rho
    v = U[start_i:end_i, start_j:end_j, 2] * inv_rho
    E_total = U[start_i:end_i, start_j:end_j, 3] * inv_rho
    kinetic_energy = 0.5 * (u * u + v * v)
    internal_energy = E_total - kinetic_energy
    p = np.maximum((GAMMA - 1) * rho * internal_energy, P_MIN)

    # Clip velocities to prevent numerical instability
    u = np.clip(u, -MAX_VAL, MAX_VAL)
    v = np.clip(v, -MAX_VAL, MAX_VAL)

    u_limited = u.copy()
    v_limited = v.copy()
    for i in range(1, u.shape[0]-1):
        for j in range(1, u.shape[1]-1):
            du_dx = minmod((u[i, j+1] - u[i, j]) / EPSILON, (u[i, j] - u[i, j-1]) / EPSILON)
            du_dy = minmod((u[i+1, j] - u[i, j]) / EPSILON, (u[i, j] - u[i-1, j]) / EPSILON)
            dv_dx = minmod((v[i, j+1] - v[i, j]) / EPSILON, (v[i, j] - v[i, j-1]) / EPSILON)
            dv_dy = minmod((v[i+1, j] - v[i, j]) / EPSILON, (v[i, j] - v[i-1, j]) / EPSILON)
            u_limited[i, j] += 0.5 * (du_dx + du_dy)
            v_limited[i, j] += 0.5 * (dv_dx + dv_dy)
    return rho, u_limited, v_limited, p

@jit(nopython=True, cache=True, fastmath=True)
def rusanov_riemann_solver_limited(Q_L, Q_R, n_vec, EPSILON, P_MIN, RHO_MIN, GAMMA, MAX_VAL, ny, nx):
    F = np.zeros((ny, 4), dtype=np.float64)
    for i in prange(ny):
        rho_L, u_L, v_L, p_L = cons_to_prim_limited(Q_L, EPSILON, P_MIN, RHO_MIN, GAMMA, MAX_VAL, i, i+1, 0, nx)
        rho_R, u_R, v_R, p_R = cons_to_prim_limited(Q_R, EPSILON, P_MIN, RHO_MIN, GAMMA, MAX_VAL, i, i+1, 0, nx)
        q_L = u_L[0, 0] * n_vec[0] + v_L[0, 0] * n_vec[1]
        q_R = u_R[0, 0] * n_vec[0] + v_R[0, 0] * n_vec[1]
        a_L = np.sqrt(max(GAMMA * p_L[0, 0] / rho_L[0, 0], EPSILON))
        a_R = np.sqrt(max(GAMMA * p_R[0, 0] / rho_R[0, 0], EPSILON))

        F_L = np.array([
            rho_L[0, 0] * q_L,
            rho_L[0, 0] * u_L[0, 0] * q_L + p_L[0, 0] * n_vec[0],
            rho_L[0, 0] * v_L[0, 0] * q_L + p_L[0, 0] * n_vec[1],
            (Q_L[i, 0, 3] + p_L[0, 0]) * q_L
        ])
        F_R = np.array([
            rho_R[0, 0] * q_R,
            rho_R[0, 0] * u_R[0, 0] * q_R + p_R[0, 0] * n_vec[0],
            rho_R[0, 0] * v_R[0, 0] * q_R + p_R[0, 0] * n_vec[1],
            (Q_R[i, 0, 3] + p_R[0, 0]) * q_R
        ])

        lambda_max = max(abs(q_L) + a_L, abs(q_R) + a_R)
        for k in range(4):
            F[i, k] = 0.5 * (F_L[k] + F_R[k]) - 0.5 * lambda_max * (Q_R[i, 0, k] - Q_L[i, 0, k])
    return np.clip(F, -MAX_VAL, MAX_VAL)

@jit(nopython=True, cache=True, fastmath=True)
def apply_cavity_bcs_improved(U, cavity_mask, RHO_INF, u_inf, P_INF, GAMMA, EPSILON, NG):
    E_inf = P_INF / (GAMMA - 1) + 0.5 * RHO_INF * u_inf * u_inf
    for j in range(U.shape[1]):
        for k in range(4):
            U[:NG, j, k] = np.array([RHO_INF, RHO_INF * u_inf, 0.0, E_inf])[k]
            U[-NG:, j, k] = U[-NG-1, j, k]
    for i in range(U.shape[0]):
        for k in range(4):
            U[i, -NG:, k] = np.array([RHO_INF, RHO_INF * u_inf, 0.0, E_inf])[k]
            U[i, :NG, k] = U[i, 2*NG-1, k]
            U[i, :NG, 2] = -U[i, 2*NG-1, 2]
    for i in prange(U.shape[0]):
        for j in prange(U.shape[1]):
            if cavity_mask[i, j] > 0.5:
                U[i, j, 0] = RHO_INF
                U[i, j, 1] = 0.0
                U[i, j, 2] = 0.0
                U[i, j, 3] = P_INF / (GAMMA - 1)
    return U

@jit(nopython=True, cache=True, fastmath=True)
def compute_dt_stable(U, dx, dy, CFL, EPSILON, GAMMA, NG, RHO_MIN, P_MIN, MAX_VAL):
    rho, u, v, p = cons_to_prim_limited(U, EPSILON, P_MIN, RHO_MIN, GAMMA, MAX_VAL, NG, U.shape[0]-NG, NG, U.shape[1]-NG)
    c = np.sqrt(np.maximum(GAMMA * p / (rho + EPSILON), EPSILON))
    u_max = np.max(np.abs(u) + c)
    v_max = np.max(np.abs(v) + c)
    dt_x = dx / (u_max + EPSILON)
    dt_y = dy / (v_max + EPSILON)
    return CFL * min(dt_x, dt_y)

@jit(nopython=True, cache=True, fastmath=True)
def compute_vorticity_fast(u, v, dx, dy, NG):
    ny, nx = u.shape
    vorticity = np.zeros_like(u)
    dx_inv = 0.5 / dx
    dy_inv = 0.5 / dy
    for i in prange(NG, ny-NG):
        for j in prange(NG, nx-NG):
            dvdx = (v[i, j+1] - v[i, j-1]) * dx_inv
            dudy = (u[i+1, j] - u[i-1, j]) * dy_inv
            vorticity[i, j] = dvdx - dudy
    return vorticity

@jit(nopython=True, cache=True, fastmath=True)
def simulation_step_optimized(U, dt, cavity_mask, dx, dy, EPSILON, P_MIN, RHO_MIN, GAMMA, MAX_VAL, NG, RHO_INF, u_inf, P_INF, ny, nx, ARTIFICIAL_VISCOSITY):
    # Ensure dx and dy are non-zero
    dx = max(dx, EPSILON)
    dy = max(dy, EPSILON)
    
    U_new = U.copy()
    U_bcs = apply_cavity_bcs_improved(U.copy(), cavity_mask, RHO_INF, u_inf, P_INF, GAMMA, EPSILON, NG)
    
    # Check for invalid values in U_bcs
    if np.any(~np.isfinite(U_bcs)):
        logger.error("Non-finite values detected in U_bcs")
        raise ValueError("Non-finite values in U_bcs")
    
    rho, u, v, p = cons_to_prim_limited(U_bcs, EPSILON, P_MIN, RHO_MIN, GAMMA, MAX_VAL, 0, U.shape[0], 0, U.shape[1])
    
    # Check for invalid values in primitive variables
    if np.any(~np.isfinite(rho)) or np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)) or np.any(~np.isfinite(p)):
        logger.error("Non-finite values detected in rho, u, v, or p")
        raise ValueError("Non-finite values in primitive variables")
    
    Q = np.zeros_like(U_bcs)
    Q[:, :, 0] = rho
    Q[:, :, 1] = u
    Q[:, :, 2] = v
    Q[:, :, 3] = p

    F = np.zeros((nx+1, ny, 4), dtype=np.float64)
    for i in prange(nx+1):
        ii = i + NG
        F[i, :, :] = rusanov_riemann_solver_limited(Q[ii-1:ii, :, :], Q[ii:ii+1, :, :], np.array([1.0, 0.0]), EPSILON, P_MIN, RHO_MIN, GAMMA, MAX_VAL, ny, nx)
    G = np.zeros((nx, ny+1, 4), dtype=np.float64)
    for j in prange(ny+1):
        jj = j + NG
        G[:, j, :] = rusanov_riemann_solver_limited(Q[:, jj-1:jj, :], Q[:, jj:jj+1, :], np.array([0.0, 1.0]), EPSILON, P_MIN, RHO_MIN, GAMMA, MAX_VAL, ny, nx)

    for i in prange(nx):
        for j in prange(ny):
            ii, jj = i + NG, j + NG
            for k in range(4):
                dF_dx = (F[i+1, j, k] - F[i, j, k]) / dx
                dG_dy = (G[i, j+1, k] - G[i, j, k]) / dy
                U_new[ii, jj, k] = U_bcs[ii, jj, k] - dt * (dF_dx + dG_dy)

                # Add artificial viscosity
                if k == 1 or k == 2:
                    d2udx2 = (U_bcs[ii, jj+1, k] - 2*U_bcs[ii, jj, k] + U_bcs[ii, jj-1, k]) / (dx * dx)
                    d2udy2 = (U_bcs[ii+1, jj, k] - 2*U_bcs[ii, jj, k] + U_bcs[ii-1, jj, k]) / (dy * dy)
                    U_new[ii, jj, k] += dt * ARTIFICIAL_VISCOSITY * (d2udx2 + d2udy2)

    return apply_cavity_bcs_improved(U_new, cavity_mask, RHO_INF, u_inf, P_INF, GAMMA, EPSILON, NG)

class CavityFlowSolver:
    def __init__(self, config):
        self.config = config
        self.setup_grid()
        self.setup_boundary_masks()
        self.initialize_fields()
        self.step = 0
        self.energy_history = []
        self.times = []

    def setup_grid(self):
        cfg = self.config
        self.x = np.linspace(-cfg.NG * cfg.dx, cfg.DOMAIN_WIDTH + cfg.NG * cfg.dx, cfg.NX + 2 * cfg.NG)
        self.y = np.linspace(-cfg.NG * cfg.dy, cfg.DOMAIN_HEIGHT + cfg.NG * cfg.dy, cfg.NY + 2 * cfg.NG)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def setup_boundary_masks(self):
        cfg = self.config
        self.cavity_mask = ((self.X >= 0.5) & (self.X <= 0.5 + cfg.CAVITY_LENGTH) & (self.Y <= cfg.CAVITY_DEPTH)).astype(np.float64)
        sigma_x = 3.0 * cfg.dx
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if not self.cavity_mask[i, j] and 0.5 <= self.X[i, j] <= 0.5 + cfg.CAVITY_LENGTH and self.Y[i, j] > cfg.CAVITY_DEPTH:
                    dist_y = self.Y[i, j] - cfg.CAVITY_DEPTH
                    if dist_y < 3.0 * sigma_x:
                        self.cavity_mask[i, j] = np.exp(-(dist_y / sigma_x)**2)

    def initialize_fields(self):
        cfg = self.config
        dtype = np.float32 if cfg.MEMORY_EFFICIENT else np.float64
        self.U = np.zeros((cfg.NX + 2*cfg.NG, cfg.NY + 2*cfg.NG, 4), dtype=dtype, order='C')
        self.U = prim_to_cons_stable(
            np.ones_like(self.U[:, :, 0]) * cfg.RHO_INF,
            np.ones_like(self.U[:, :, 0]) * cfg.u_inf,
            np.zeros_like(self.U[:, :, 0]),
            np.ones_like(self.U[:, :, 0]) * cfg.P_INF,
            cfg.EPSILON, cfg.GAMMA, cfg.RHO_MIN
        )
        self.U = apply_cavity_bcs_improved(self.U, self.cavity_mask, cfg.RHO_INF, cfg.u_inf, cfg.P_INF, cfg.GAMMA, cfg.EPSILON, cfg.NG)

    def compute_energy(self):
        rho, u, v, p = cons_to_prim_limited(self.U, self.config.EPSILON, self.config.P_MIN, self.config.RHO_MIN, self.config.GAMMA, self.config.MAX_VAL, 0, self.U.shape[0], 0, self.U.shape[1])
        return 0.5 * rho * (u**2 + v**2)

    def time_step(self):
        cfg = self.config
        dt = compute_dt_stable(self.U, cfg.dx, cfg.dy, cfg.CFL, cfg.EPSILON, cfg.GAMMA, cfg.NG, cfg.RHO_MIN, cfg.P_MIN, cfg.MAX_VAL)
        if not np.isfinite(dt) or dt <= 0.0:
            logger.error(f"Invalid timestep dt={dt} at step {self.step}")
            return 0.0
        self.U = simulation_step_optimized(
            self.U, dt, self.cavity_mask, cfg.dx, cfg.dy, cfg.EPSILON, cfg.P_MIN, cfg.RHO_MIN,
            cfg.GAMMA, cfg.MAX_VAL, cfg.NG, cfg.RHO_INF, cfg.u_inf, cfg.P_INF, cfg.NY, cfg.NX, cfg.ARTIFICIAL_VISCOSITY
        )
        energy = self.compute_energy()
        energy_mean = np.nanmean(energy)
        self.energy_history.append((self.step, energy_mean))
        self.times.append(self.step * dt)
        logger.info(f"Step {self.step}: Mean kinetic energy = {energy_mean:.3f}")
        self.step += 1
        return dt

class CavityVisualizer:
    def __init__(self, config):
        self.config = config
        self.output_path = os.path.join(config.OUTPUT_DIR, "frames")
        self.data_path = config.OUTPUT_DIR
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=2)

    def save_data_to_hdf5(self, solver, step, current_time):
        cfg = self.config
        try:
            with h5py.File(os.path.join(self.data_path, cfg.HDF5_FILE), 'a') as f:
                group_name = f"step_{step:06d}"
                if group_name not in f:
                    group = f.create_group(group_name)
                    group.attrs['time'] = float(current_time)
                    group.create_dataset('U', data=solver.U, compression='gzip', compression_opts=4)
                    group.create_dataset('X', data=solver.X, compression='gzip', compression_opts=4)
                    group.create_dataset('Y', data=solver.Y, compression='gzip', compression_opts=4)
                logger.info(f"Saved data for step {step} to HDF5")
        except Exception as e:
            logger.error(f"Error saving HDF5 data for step {step}: {e}")

    def generate_visualizations_from_hdf5(self):
        cfg = self.config
        hdf5_path = os.path.join(self.data_path, cfg.HDF5_FILE)
        if not os.path.exists(hdf5_path):
            logger.warning(f"HDF5 file {hdf5_path} does not exist. Skipping frame generation.")
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
                rho, u, v, p = cons_to_prim_limited(U, cfg.EPSILON, cfg.P_MIN, cfg.RHO_MIN, cfg.GAMMA, cfg.MAX_VAL, 0, U.shape[0], 0, U.shape[1])
                vel_mag = np.sqrt(u**2 + v**2)
                vorticity = compute_vorticity_fast(u, v, cfg.dx, cfg.dy, cfg.NG)

                fields = [
                    (vel_mag, 'Velocity Magnitude', 'viridis', f"velocity_{step:06d}.png", np.nanmax(vel_mag)*0.9, 31, 'max'),
                    (rho, 'Density', 'plasma', f"density_{step:06d}.png", np.nanmax(rho)*0.9, 31, 'max'),
                    (vorticity, 'Vorticity', 'inferno', f"vorticity_{step:06d}.png", min(np.nanmax(np.abs(vorticity)), 15.0), 51, 'both'),
                    (p, 'Pressure', 'magma', f"pressure_{step:06d}.png", np.nanmax(p)*0.9, 31, 'max')
                ]
                for field, title, cmap, filename, max_val, levels, extend in fields:
                    try:
                        fig = plt.figure(figsize=(12, 6))
                        ax = fig.add_subplot(111)
                        levels_array = np.linspace(-max_val, max_val, levels) if 'Vorticity' in title else np.linspace(0, max_val, levels)
                        cf = ax.contourf(X[2:-2:2, 2:-2:2], Y[2:-2:2, 2:-2:2], field[2:-2:2, 2:-2:2], levels=levels_array, cmap=cmap, extend=extend)
                        plt.colorbar(cf, ax=ax, label=title, shrink=0.8)
                        skip = max(15, min(X.shape[0], X.shape[1]) // 15)
                        ax.quiver(X[2:-2:skip, 2:-2:skip],Y[2:-2:skip, 2:-2:skip],u[2:-2:skip, 2:-2:skip],v[2:-2:skip, 2:-2:skip],color='lightgray', scale=40, alpha=0.3)
                        cavity_patch = patches.Rectangle((0.5, 0.0), cfg.CAVITY_LENGTH, cfg.CAVITY_DEPTH, facecolor='black', edgecolor='gold', linewidth=1.5)
                        ax.add_patch(cavity_patch)
                        ax.set_xlim(-cfg.dx*cfg.NG, cfg.DOMAIN_WIDTH + cfg.dx*cfg.NG)
                        ax.set_ylim(-cfg.dy*cfg.NG, cfg.DOMAIN_HEIGHT + cfg.dy*cfg.NG)
                        ax.set_aspect('equal')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_title(f'{title}, Mach={cfg.MACH_INF:.1f}, t={current_time:.3f}s')
                        ax.grid(True, alpha=0.2)
                        max_field = np.nanmax(np.abs(field))
                        mean_field = np.nanmean(field)
                        fig.text(0.02, 0.02, f'Max {title}: {max_field:.3f} | Mean {title}: {mean_field:.3f}', fontsize=8, color='white')
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.output_path, filename), dpi=cfg.DPI, bbox_inches='tight')
                        plt.close(fig)
                    except Exception as e:
                        logger.error(f"Error plotting {title} frame {step}: {e}")
                        plt.close('all')
                if cfg.MEMORY_EFFICIENT:
                    gc.collect()

    def plot_energy_history(self, solver):
        cfg = self.config
        try:
            if not solver.energy_history:
                logger.warning("No energy history data to plot.")
                return
            steps, energies = zip(*solver.energy_history)
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121)
            ax1.semilogx(steps, energies, label='Mean Kinetic Energy', color='cyan')
            ax1.set_xlabel('Steps (log scale)')
            ax1.set_ylabel('Mean Kinetic Energy')
            ax1.set_title(f'Kinetic Energy History, Mach={cfg.MACH_INF:.1f}')
            ax1.grid(True, which='both', alpha=0.3)
            ax1.legend()

            ax2 = fig.add_subplot(122)
            interval = 200
            energy_intervals = [np.mean([e for _, e in solver.energy_history[i:i+interval]]) for i in range(0, len(solver.energy_history), interval)]
            interval_steps = [s for s, _ in solver.energy_history[::interval]]
            ax2.bar(interval_steps, energy_intervals, color='orange', alpha=0.7, width=interval*0.8)
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Mean Kinetic Energy (Averaged over 200 Steps)')
            ax2.set_title('Energy Intervals')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = os.path.join(self.data_path, "energy_history.png")
            plt.savefig(filename, dpi=cfg.DPI, bbox_inches='tight')
            plt.close(fig)
            if cfg.MEMORY_EFFICIENT:
                gc.collect()
            logger.info("Saved energy history plot")
        except Exception as e:
            logger.error(f"Error plotting energy history: {e}")
            plt.close('all')

    def cleanup(self):
        self.executor.shutdown(wait=False)
        logger.info("Visualizer cleanup completed")

def prim_to_cons_stable(rho, u, v, p, EPSILON, GAMMA, RHO_MIN):
    rho = np.maximum(rho, RHO_MIN)
    mom_x = rho * u
    mom_y = rho * v
    kinetic_energy = 0.5 * rho * (u*u + v*v)
    internal_energy = p / (GAMMA - 1)
    E_total = internal_energy + kinetic_energy
    U = np.zeros((rho.shape[0], rho.shape[1], 4), dtype=np.float64)
    U[:, :, 0] = rho
    U[:, :, 1] = mom_x
    U[:, :, 2] = mom_y
    U[:, :, 3] = E_total
    return U

def monitor_simulation_health(solver, step):
    cfg = solver.config
    rho, u, v, p = cons_to_prim_limited(solver.U, cfg.EPSILON, cfg.P_MIN, cfg.RHO_MIN, cfg.GAMMA, cfg.MAX_VAL, 0, solver.U.shape[0], 0, solver.U.shape[1])
    if np.any(~np.isfinite(solver.U)) or np.any(rho < cfg.RHO_MIN) or np.any(p < cfg.P_MIN):
        logger.error(f"Non-finite values or invalid density/pressure at step {step}")
        return False
    vel_max = max(np.max(np.abs(u)), np.max(np.abs(v)))
    if vel_max > cfg.MAX_VAL:
        logger.warning(f"High velocity {vel_max:.3f} at step {step}")
        return False
    return True

def main():
    config = CavityFlowConfig()
    logger.info("🚀 Starting Optimized Supersonic Cavity Flow Solver...")
    logger.info("🎯 Target: Complete simulation in under 1.5 hours with >100 steps/sec")

    solver = CavityFlowSolver(config)
    visualizer = CavityVisualizer(config)
    t = 0.0
    step = 0
    sim_start_time = time.time()
    CFL = config.CFL

    visualizer.save_data_to_hdf5(solver, step, t)

    try:
        with tqdm(total=config.T_FINAL, desc="Simulation Progress", unit="s",
                  bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f}s [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            while t < config.T_FINAL:
                dt = solver.time_step()
                if dt <= 0.0:
                    logger.error("Simulation stopped due to invalid timestep.")
                    break
                t += dt

                if step % 50 == 0:
                    if not monitor_simulation_health(solver, step):
                        CFL = max(CFL * 0.8, config.CFL_MIN)
                        logger.warning(f"Reducing CFL to {CFL:.3f}")
                        if CFL == config.CFL_MIN:
                            logger.error("Minimum CFL reached. Stopping simulation.")
                            break
                        config.CFL = CFL

                if step % config.SAVE_INTERVAL == 0:
                    visualizer.save_data_to_hdf5(solver, step, t)
                    elapsed = time.time() - sim_start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    remaining_time = (config.T_FINAL - t) / (t / elapsed) if t > 0 else float('inf')
                    logger.info(f"Step: {step:6d}, Time: {t:8.4f}s, dt: {dt:.2e}s, Speed: {steps_per_sec:6.1f} steps/s, ETA: {remaining_time/60:.1f}min")
                    if config.MEMORY_EFFICIENT:
                        memory_usage = psutil.Process().memory_info().rss / (1024*1024)
                        logger.info(f"Memory usage: {memory_usage:.1f} MB")

                step += 1
                pbar.update(dt)

                if time.time() - sim_start_time > 5400:  # 1.5 hours
                    logger.warning("Time limit reached. Stopping simulation.")
                    break

    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        visualizer.generate_visualizations_from_hdf5()
        visualizer.plot_energy_history(solver)
        sim_end_time = time.time()
        total_time = sim_end_time - sim_start_time
        final_speed = step / total_time if total_time > 0 else 0
        logger.info("\n🏆 Simulation Performance Report:")
        logger.info(f"Total steps: {step}")
        logger.info(f"Final time: {t:.2f}")
        logger.info(f"Wall time: {total_time/60:.1f} minutes")
        logger.info(f"Average speed: {final_speed:.1f} steps/second")
        logger.info(f"Results saved in: {os.path.join(config.OUTPUT_DIR, config.HDF5_FILE)}")
        visualizer.cleanup()
        del solver, visualizer
        gc.collect()

if __name__ == "__main__":
    main()