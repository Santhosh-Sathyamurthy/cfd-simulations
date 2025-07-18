# MIT License
# Copyright (c) 2025 Santhosh S
# See LICENSE file for full license text.
#
# Description:
# This script is a robust, multithreaded alternative to the MPI-based version.
# It avoids the complexities of GridapDistributed and PartitionedArrays by using
# Julia's native shared-memory parallelism, which is simpler and more stable
# for single-machine execution.
#
# The solver's performance is accelerated by the multithreaded BLAS library.

# --- HOW TO RUN ---
# This script uses Julia's native multithreading.
# 1. Open your terminal.
# 2. Run the script with the `--threads` flag, specifying the number of cores to use.
#
#    julia --threads 8 v2_parallelization_fixed.jl
#
#    Replace '8' with the number of CPU cores you want to use.

# --- Load Packages ---
using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using GridapGmsh
using Printf
using Statistics
using LinearAlgebra
using LineSearches  # Add this import for BackTracking

# --- Global Simulation Parameters ---
const L = 1.0
const R_cylinder = L / 15
const V_inf = 1.0
const cylinder_center = (2.5 * L, 0.0)
const T_FINAL = 10.0
const DT = 0.02
const N_STEPS = Int(T_FINAL / DT)
const SAVE_INTERVAL_VTK = 10
const BASE_OUTPUT_DIR = "optimized_cylinder_flow_threaded"
const H_FAR = 0.08
const H_NEAR = 0.008
const RE_VALUES = [100.0, 200.0, 400.0]
const MEMORY_THRESHOLD_GB = 7.0 # RAM cap

"""
The main simulation function.
"""
function run_simulation()
    # --- System and Memory Configuration ---
    # Set the number of threads the BLAS library should use.
    # By default, it will use the number of threads Julia was started with.
    num_threads = BLAS.get_num_threads()
    println("Running with $(num_threads) threads.")

    function check_memory_and_exit()
        available_mem_gb = Sys.free_memory() / 1024^3
        if available_mem_gb < MEMORY_THRESHOLD_GB
            @warn "Available memory ($(round(available_mem_gb, digits=2)) GB) is below the threshold of $(MEMORY_THRESHOLD_GB) GB. Exiting gracefully."
            exit(1)
        end
    end

    # --- Mesh Generation ---
    println("\n" * "="^60)
    println("OPTIMIZED MULTITHREADED CYLINDER FLOW SIMULATION")
    println("="^60)
    println("Julia version: $(VERSION)")
    println("BLAS vendor: $(BLAS.vendor()) with $(num_threads) threads")
    println("Reynolds numbers to be simulated: $(RE_VALUES)")
    println("="^60)

    mkpath(BASE_OUTPUT_DIR)
    for Re in RE_VALUES
        mkpath(joinpath(BASE_OUTPUT_DIR, "Re_$(Re)"))
    end

    println("\n Generating mesh...")
    msh_file = joinpath(BASE_OUTPUT_DIR, "cylinder_optimized.msh")
    geo_content = """
    L = $(L); R = $(R_cylinder);
    x_min = -$(L); x_max = $(6*L); y_min = -$(1.5*L); y_max = $(1.5*L);
    cx = $(cylinder_center[1]); cy = $(cylinder_center[2]);
    h_far = $(H_FAR); h_near = $(H_NEAR);
    Point(1) = {x_min, y_min, 0, h_far}; Point(2) = {x_max, y_min, 0, h_far};
    Point(3) = {x_max, y_max, 0, h_far}; Point(4) = {x_min, y_max, 0, h_far};
    Point(5) = {cx, cy, 0, h_near}; Point(6) = {cx + R, cy, 0, h_near};
    Point(7) = {cx, cy + R, 0, h_near}; Point(8) = {cx - R, cy, 0, h_near};
    Point(9) = {cx, cy - R, 0, h_near};
    Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
    Circle(5) = {6, 5, 7}; Circle(6) = {7, 5, 8};
    Circle(7) = {8, 5, 9}; Circle(8) = {9, 5, 6};
    Line Loop(1) = {1, 2, 3, 4}; Line Loop(2) = {5, 6, 7, 8};
    Plane Surface(1) = {1, 2};
    Physical Line("inlet") = {4}; Physical Line("outlet") = {2};
    Physical Line("walls") = {1, 3}; Physical Line("cylinder") = {5, 6, 7, 8};
    Physical Surface("domain") = {1};
    Mesh.Algorithm = 6; Mesh.OptimizeNetgen = 1;
    """
    open(joinpath(BASE_OUTPUT_DIR, "cylinder.geo"), "w") do file
        write(file, geo_content)
    end
    try
        run(`gmsh -2 $(joinpath(BASE_OUTPUT_DIR, "cylinder.geo")) -o $msh_file`)
    catch e
        error("Gmsh execution failed. Make sure Gmsh is installed and in your system's PATH. Error: $e")
    end

    model = GmshDiscreteModel(msh_file)
    println(" Mesh generation complete.")

    # --- Simulation Loop ---
    for Re in RE_VALUES
        println("\n" * "="^50)
        println("Processing Re = $Re")
        println("="^50)

        order_u = 2
        order_p = 1
        reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)
        reffe_p = ReferenceFE(lagrangian, Float64, order_p)

        V = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["inlet", "walls", "cylinder"])
        Q = TestFESpace(model, reffe_p, conformity=:H1, constraint=:zeromean)

        u_inlet(x, t) = VectorValue(V_inf, 0.0)
        u_walls(x, t) = VectorValue(0.0, 0.0)
        u_cylinder(x, t) = VectorValue(0.0, 0.0)

        U = TransientTrialFESpace(V, [u_inlet, u_walls, u_cylinder])
        P = TransientTrialFESpace(Q)

        Y = MultiFieldFESpace([V, Q])
        X = TransientMultiFieldFESpace([U, P])

        ν = V_inf * L / Re
        degree = 2 * order_u
        Ω = Triangulation(model)
        dΩ = Measure(Ω, degree)

        # Define the mass matrix form for the velocity component only
        mass_u(∂t_u, v) = ∫( ∂t_u ⋅ v )dΩ
        
        # Define the residual form
        res((u, p), (v, q)) = ∫( ν*(∇(u)⊙∇(v)) + (∇(u)'⋅u)⋅v - (∇⋅v)*p + q*(∇⋅u) )dΩ
        
        # Define the jacobian form
        jac((u, p), (δu, δp), (v, q)) = ∫( ν*(∇(δu)⊙∇(v)) + (∇(δu)'⋅u)⋅v + (∇(u)'⋅δu)⋅v - (∇⋅v)*δp + q*(∇⋅δu) )dΩ
        
        # Define the mass jacobian (jacobian with respect to time derivative)
        jac_t((u, p), (δu, δp), (v, q)) = ∫( δu ⋅ v )dΩ

        # Create the transient FE operator with the correct syntax
        op = TransientFEOperator(res, jac, jac_t, X, Y)
        
        nls = NLSolver(show_trace=false, method=:newton, linesearch=LineSearches.BackTracking(), rel_tol=1e-6, abs_tol=1e-8, max_iter=10)
        solver = FESolver(nls)

        xh0 = interpolate_everywhere([VectorValue(0.0, 0.0), 0.0], X(0.0))

        println("Starting time integration for Re = $Re...")
        t0 = 0.0
        time_stepper = ThetaMethod(solver, DT, 0.5)
        sol_t = solve(time_stepper, op, xh0, t0, T_FINAL)

        re_output_dir = joinpath(BASE_OUTPUT_DIR, "Re_$(Re)")
        step_count = 0
        start_time = time()

        # Create a .pvd file to group the time series data.
        createpvd(joinpath(re_output_dir, "results_Re_$(Re)")) do pvd
            for (xh_t, t_n) in sol_t
                step_count += 1
                check_memory_and_exit()
                if step_count % SAVE_INTERVAL_VTK == 0
                    println("  -> Re=$Re, Step=$step_count/$(N_STEPS), Time=$(@sprintf("%.2f", t_n))s")
                    uh, ph = xh_t
                    pvd[t_n] = create_vtk_file(Ω, joinpath(re_output_dir, "results_$(step_count)"), cellfields=["uh" => uh, "ph" => ph])
                end
            end
        end

        total_time = time() - start_time
        println("\nRe = $Re completed:")
        println("  Total time: $(@sprintf("%.2f", total_time)) seconds")
        println("  Performance: $(@sprintf("%.2f", step_count / total_time)) steps/second")
        println("  Results saved in: $(re_output_dir)")
    end

    println("\n" * "="^60)
    println("ALL SIMULATIONS COMPLETED SUCCESSFULLY!")
    println("="^60)
end

# --- Main Execution Block ---
try
    run_simulation()
catch e
    println("A critical error occurred: $e")
    rethrow(e)
end