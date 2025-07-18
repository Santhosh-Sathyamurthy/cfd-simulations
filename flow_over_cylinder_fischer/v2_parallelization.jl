# MIT License
# Copyright (c) 2025 Santhosh S
# See LICENSE file for full license text.

using Pkg
using LinearAlgebra
using Distributed
using Printf
using Statistics
using ProgressMeter
using SparseArrays
using Plots
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Fields
using GridapGmsh

# Force MKL usage
ENV["JULIA_MKL_LIBRARY"] = "mkl_rt"
try
    @eval using MKL
    println("✓ MKL loaded successfully")
catch e
    println("MKL not installed. Installing now...")
    try
        Pkg.add("MKL")
        @eval using MKL
        Pkg.precompile()
        println("✓ MKL installed and loaded")
    catch
        println("WARNING: Failed to install MKL. Falling back to OpenBLAS.")
    end
end

# Verify BLAS vendor
println("✓ BLAS vendor: ", BLAS.vendor())
if BLAS.vendor() != :mkl
    @warn "MKL not active for BLAS operations. Using $(BLAS.vendor()) instead."
end

# Set optimal BLAS threads
optimal_threads = min(Sys.CPU_THREADS ÷ 2, 8)
BLAS.set_num_threads(optimal_threads)
println("✓ BLAS threads set to: ", BLAS.get_num_threads())

# System diagnostics
println("System specs:")
println("  CPU cores: $(Sys.CPU_THREADS)")
println("  Available memory: $(round(Sys.total_memory() / 1024^3, digits=1)) GB")

# Parallel setup
const N_WORKERS = min(8, Sys.CPU_THREADS - 2)
if nprocs() == 1 && N_WORKERS > 1
    addprocs(N_WORKERS; exeflags="--project")
    println("Added $N_WORKERS worker processes")
end

# Load packages on all workers
@everywhere begin
    using Gridap
    using Gridap.Geometry
    using Gridap.ReferenceFEs
    using Gridap.FESpaces
    using Gridap.MultiField
    using Gridap.Fields
    using LinearAlgebra
    using Printf
end

# Simulation parameters
const L = 1.0
const R_cylinder = L/15
const V_inf = 1.0
const cylinder_center = (2.5*L, 0.0)
const T_TOTAL = 30.0
const DT = 0.01
const N_STEPS = Int(T_TOTAL / DT)
const SAVE_INTERVAL = 5
const H_FAR = 0.08
const H_NEAR = 0.008
const RE_VALUES = [100.0, 200.0, 400.0, 800.0, 1600.0]

# Output management
function setup_output_directories()
    base_dir = joinpath(pwd(), "optimized_cylinder_flow")
    isdir(base_dir) || mkdir(base_dir)
    for subdir in ["fields", "analysis"]
        full_path = joinpath(base_dir, subdir)
        isdir(full_path) || mkdir(full_path)
    end
    return base_dir
end

# Mesh generation
function create_optimized_mesh(h_far=H_FAR, h_near=H_NEAR)
    geo_content = """
    L = $(L);
    R = $(R_cylinder);
    x_min = -$(L);
    x_max = $(6*L);
    y_min = -$(1.5*L);
    y_max = $(1.5*L);
    cx = $(cylinder_center[1]);
    cy = $(cylinder_center[2]);
    h_far = $(h_far);
    h_near = $(h_near);
    
    Point(1) = {x_min, y_min, 0, h_far};
    Point(2) = {x_max, y_min, 0, h_far};
    Point(3) = {x_max, y_max, 0, h_far};
    Point(4) = {x_min, y_max, 0, h_far};
    Point(5) = {cx, cy, 0, h_near};
    Point(6) = {cx + R, cy, 0, h_near};
    Point(7) = {cx, cy + R, 0, h_near};
    Point(8) = {cx - R, cy, 0, h_near};
    Point(9) = {cx, cy - R, 0, h_near};
    
    Line(1) = {1, 2};
    Line(2) = {2, 3};
    Line(3) = {3, 4};
    Line(4) = {4, 1};
    Circle(5) = {6, 5, 7};
    Circle(6) = {7, 5, 8};
    Circle(7) = {8, 5, 9};
    Circle(8) = {9, 5, 6};
    
    Line Loop(1) = {1, 2, 3, 4};
    Line Loop(2) = {5, 6, 7, 8};
    Plane Surface(1) = {1, 2};
    
    Physical Line("inlet") = {4};
    Physical Line("outlet") = {2};
    Physical Line("walls") = {1, 3};
    Physical Line("cylinder") = {5, 6, 7, 8};
    Physical Surface("domain") = {1};
    
    Mesh.Algorithm = 6;
    Mesh.CharacteristicLengthMin = $(h_near);
    Mesh.CharacteristicLengthMax = $(h_far);
    Mesh.OptimizeNetgen = 1;
    """
    
    geo_file = joinpath(pwd(), "cylinder_optimized.geo")
    msh_file = joinpath(pwd(), "cylinder_optimized.msh")
    
    try
        open(geo_file, "w") do file
            write(file, geo_content)
        end
        run(`gmsh -2 $geo_file -o $msh_file`)
        model = GmshDiscreteModel(msh_file)
        return model
    catch e
        error("Failed to create mesh: $e")
    end
end

# FIXED: Corrected solver setup with proper boundary condition handling
function create_optimized_solver(model, Re)
    ν = V_inf * L / Re
    
    # Use P2-P1 Taylor-Hood elements for stability
    order_u = 2
    order_p = 1
    
    # Reference finite elements
    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)
    reffe_p = ReferenceFE(lagrangian, Float64, order_p)
    
    # Test spaces - velocity with Dirichlet conditions
    V = TestFESpace(model, reffe_u, 
                    conformity=:H1, 
                    dirichlet_tags=["inlet", "walls", "cylinder"])
    
    # Test space for pressure - no boundary conditions
    Q = TestFESpace(model, reffe_p, conformity=:H1)
    
    # Boundary condition values
    u_inlet = VectorValue(V_inf, 0.0)
    u_walls = VectorValue(0.0, 0.0)
    u_cylinder = VectorValue(0.0, 0.0)
    
    # Trial spaces with boundary conditions
    U = TrialFESpace(V, [u_inlet, u_walls, u_cylinder])
    P = TrialFESpace(Q)  # No constraints on pressure
    
    # Create multi-field spaces
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    # Integration measures
    degree = 2 * max(order_u, order_p)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    # Stabilization parameters
    h = H_NEAR
    τ_supg = h / (2 * V_inf)
    τ_pspg = h^2 / (4 * ν + 2 * V_inf * h)
    
    return X, Y, dΩ, ν, τ_supg, τ_pspg
end

# FIXED: Corrected Stokes solver with proper variable unpacking
function solve_stokes_optimized(X, Y, dΩ, ν, τ_pspg)
    # Bilinear form for Stokes equations
    function a_stokes((u, p), (v, q))
        # Viscous term
        viscous = ν * (∇(u) ⊙ ∇(v))
        # Pressure terms
        pressure = - (∇ ⋅ v) * p + q * (∇ ⋅ u)
        # PSPG stabilization
        stabilization = τ_pspg * (∇(p) ⊙ ∇(q))
        
        return ∫(viscous + pressure + stabilization) * dΩ
    end
    
    # Linear form (zero for Stokes)
    function b_stokes((v, q))
        return ∫(0.0 * v[1] + 0.0 * q) * dΩ
    end
    
    try
        # Create and solve the linear system
        op = AffineFEOperator(a_stokes, b_stokes, X, Y)
        xh = solve(op)
        
        # Extract velocity and pressure
        uh, ph = xh
        return uh, ph, true
    catch e
        @warn "Stokes solver failed: $e"
        return nothing, nothing, false
    end
end

# FIXED: Corrected Navier-Stokes solver
function solve_ns_optimized(X, Y, dΩ, ν, τ_supg, τ_pspg, dt, u_prev, p_prev)
    # Bilinear form for Navier-Stokes
    function a_ns((u, p), (v, q))
        # Time derivative term
        time_term = (u - u_prev) ⊙ v / dt
        # Viscous term
        viscous_term = ν * (∇(u) ⊙ ∇(v))
        # Convective term (linearized)
        convective_term = ((u_prev ⋅ ∇(u)) ⊙ v)
        # Pressure terms
        pressure_term = - (∇ ⋅ v) * p + q * (∇ ⋅ u)
        
        # Stabilization terms
        if ν < 0.01  # High Reynolds number
            stab_term = τ_supg * ((u_prev ⋅ ∇(u)) ⊙ (u_prev ⋅ ∇(v))) + 
                       τ_pspg * (∇(p) ⊙ ∇(q))
        else
            stab_term = τ_pspg * (∇(p) ⊙ ∇(q))
        end
        
        return ∫(time_term + viscous_term + convective_term + pressure_term + stab_term) * dΩ
    end
    
    # Linear form
    function b_ns((v, q))
        return ∫(0.0 * v[1] + 0.0 * q) * dΩ
    end
    
    try
        op = AffineFEOperator(a_ns, b_ns, X, Y)
        xh = solve(op)
        uh, ph = xh
        return uh, ph, true
    catch e
        @warn "Navier-Stokes solver failed: $e"
        return u_prev, p_prev, false
    end
end

# Parallel field computation
@everywhere function compute_fields_optimized(x_chunk, y_range, uh, ph, cylinder_center, R_cylinder)
    nx, ny = length(x_chunk), length(y_range)
    vorticity = zeros(nx, ny)
    velocity_mag = zeros(nx, ny)
    pressure_field = zeros(nx, ny)
    cylinder_mask = zeros(Bool, nx, ny)
    
    # Pre-compute cylinder mask
    for (i, x) in enumerate(x_chunk)
        for (j, y) in enumerate(y_range)
            dist_sq = (x - cylinder_center[1])^2 + (y - cylinder_center[2])^2
            cylinder_mask[i, j] = dist_sq <= (R_cylinder * 1.05)^2
        end
    end
    
    # Compute fields
    for (i, x) in enumerate(x_chunk)
        for (j, y) in enumerate(y_range)
            if !cylinder_mask[i, j]
                try
                    point = Point(x, y)
                    u_val = uh(point)
                    p_val = ph(point)
                    
                    velocity_mag[i, j] = sqrt(u_val[1]^2 + u_val[2]^2)
                    pressure_field[i, j] = p_val
                    
                    # Compute vorticity using finite differences
                    if i > 1 && i < nx && j > 1 && j < ny
                        h = min(0.02, (x_chunk[end] - x_chunk[1]) / (nx - 1))
                        try
                            u_xp = uh(Point(x + h, y))
                            u_xm = uh(Point(x - h, y))
                            u_yp = uh(Point(x, y + h))
                            u_ym = uh(Point(x, y - h))
                            
                            dudy = (u_yp[1] - u_ym[1]) / (2*h)
                            dvdx = (u_xp[2] - u_xm[2]) / (2*h)
                            vorticity[i, j] = dvdx - dudy
                        catch
                            vorticity[i, j] = 0.0
                        end
                    end
                catch
                    vorticity[i, j] = 0.0
                    velocity_mag[i, j] = 0.0
                    pressure_field[i, j] = 0.0
                end
            else
                vorticity[i, j] = NaN
                velocity_mag[i, j] = NaN
                pressure_field[i, j] = NaN
            end
        end
    end
    
    return vorticity, velocity_mag, pressure_field
end

# Visualization function
function create_optimized_visualization(uh, ph, Re, step, base_dir)
    try
        x_range = range(-0.5*L, 5*L, length=120)
        y_range = range(-L, L, length=80)
        
        # Parallel computation of fields
        n_chunks = min(nprocs(), 4)
        x_chunks = [x_range[i:n_chunks:end] for i in 1:n_chunks]
        
        results = pmap(chunk -> compute_fields_optimized(chunk, y_range, uh, ph, cylinder_center, R_cylinder), x_chunks)
        
        # Reconstruct full arrays
        nx, ny = length(x_range), length(y_range)
        vorticity = zeros(nx, ny)
        velocity_mag = zeros(nx, ny)
        pressure_field = zeros(nx, ny)
        
        for (i, (vort, vel, press)) in enumerate(results)
            indices = i:n_chunks:nx
            vorticity[indices, :] = vort
            velocity_mag[indices, :] = vel
            pressure_field[indices, :] = press
        end
        
        # Cylinder boundary for plotting
        θ = range(0, 2π, length=50)
        cyl_x = cylinder_center[1] .+ R_cylinder * cos.(θ)
        cyl_y = cylinder_center[2] .+ R_cylinder * sin.(θ)
        
        t_current = step * DT
        theme(:dark)
        
        # Vorticity plot
        valid_vort = filter(!isnan, vec(vorticity))
        if length(valid_vort) > 0
            vort_scale = quantile(abs.(valid_vort), 0.9)
            vort_lim = max(vort_scale, 0.1)
            
            p1 = heatmap(x_range, y_range, vorticity',
                        c=:plasma, aspect_ratio=:equal, size=(800, 400),
                        clim=(-vort_lim, vort_lim), dpi=100,
                        title="Vorticity - Re = $Re, t = $(@sprintf("%.1f", t_current))")
            plot!(p1, cyl_x, cyl_y, color=:white, linewidth=2)
            
            filename = joinpath(base_dir, "fields", "vort_Re$(Re)_$(@sprintf("%04d", step)).png")
            savefig(p1, filename)
        end
        
        # Velocity magnitude plot
        p2 = heatmap(x_range, y_range, velocity_mag',
                    c=:viridis, aspect_ratio=:equal, size=(800, 400),
                    clim=(0, 1.5*V_inf), dpi=100,
                    title="Velocity - Re = $Re, t = $(@sprintf("%.1f", t_current))")
        plot!(p2, cyl_x, cyl_y, color=:white, linewidth=2)
        
        filename = joinpath(base_dir, "fields", "vel_Re$(Re)_$(@sprintf("%04d", step)).png")
        savefig(p2, filename)
        
        return true
    catch e
        @warn "Visualization failed: $e"
        return false
    end
end

# Performance monitoring
function monitor_performance(step, total_steps, start_time, Re)
    if step % (total_steps ÷ 10) == 0
        elapsed = time() - start_time
        steps_per_sec = step / elapsed
        eta = (total_steps - step) / steps_per_sec
        mem = Sys.free_memory() / 1024^3
        println("    Re=$Re: Step $step/$total_steps | $(@sprintf("%.1f", steps_per_sec)) steps/s | ETA: $(@sprintf("%.1f", eta/60)) min | Free memory: $(@sprintf("%.1f", mem)) GB")
    end
end

# Main simulation function - FIXED variable scoping
function run_optimized_simulation()
    println("\n" * "="^60)
    println("OPTIMIZED CYLINDER FLOW CFD SIMULATION")
    println("="^60)
    println("System: $(Sys.CPU_THREADS) cores, $(round(Sys.total_memory()/1024^3, digits=1)) GB RAM")
    println("Workers: $(nprocs()-1)")
    println("BLAS: $(BLAS.vendor()) with $(BLAS.get_num_threads()) threads")
    println("Reynolds numbers: $(RE_VALUES)")
    println("Time steps: $N_STEPS")
    println("="^60)
    
    base_dir = setup_output_directories()
    
    println("\nCreating optimized mesh...")
    mesh_start = time()
    model = create_optimized_mesh()
    mesh_time = time() - mesh_start
    println("Mesh created in $(@sprintf("%.2f", mesh_time)) seconds")
    
    results = Dict{Float64, Dict{String, Any}}()
    
    for (re_idx, Re) in enumerate(RE_VALUES)
        println("\n" * "="^50)
        println("Processing Re = $Re ($re_idx/$(length(RE_VALUES)))")
        println("="^50)
        
        results[Re] = Dict(
            "converged_steps" => 0,
            "total_time" => 0.0,
            "avg_step_time" => 0.0
        )
        
        # FIXED: Declare variables outside try block to fix scoping
        X, Y, dΩ, ν, τ_supg, τ_pspg = nothing, nothing, nothing, nothing, nothing, nothing
        
        # Solver setup
        println("Setting up solver...")
        solver_start = time()
        
        try
            X, Y, dΩ, ν, τ_supg, τ_pspg = create_optimized_solver(model, Re)
            solver_time = time() - solver_start
            println("Solver setup: $(@sprintf("%.2f", solver_time)) seconds")
        catch e
            println("ERROR: Solver setup failed for Re = $Re: $e")
            continue
        end
        
        # Check if solver setup was successful
        if X === nothing || Y === nothing || dΩ === nothing
            println("ERROR: Solver setup returned nothing for Re = $Re")
            continue
        end
        
        # Solve initial Stokes problem
        println("Solving Stokes...")
        stokes_start = time()
        uh, ph, stokes_ok = solve_stokes_optimized(X, Y, dΩ, ν, τ_pspg)
        stokes_time = time() - stokes_start
        
        if !stokes_ok || uh === nothing || ph === nothing
            println("ERROR: Stokes solver failed for Re = $Re")
            continue
        end
        
        println("Stokes solved in $(@sprintf("%.2f", stokes_time)) seconds")
        create_optimized_visualization(uh, ph, Re, 0, base_dir)
        
        # Time integration
        println("Starting time integration...")
        simulation_start = time()
        converged_steps = 0
        
        for step in 1:N_STEPS
            uh_new, ph_new, converged = solve_ns_optimized(X, Y, dΩ, ν, τ_supg, τ_pspg, DT, uh, ph)
            
            if converged && uh_new !== nothing && ph_new !== nothing
                uh, ph = uh_new, ph_new
                converged_steps += 1
            else
                # If solver fails, keep previous solution
                @warn "Step $step failed for Re = $Re, keeping previous solution"
            end
            
            if step % SAVE_INTERVAL == 0
                create_optimized_visualization(uh, ph, Re, step, base_dir)
            end
            
            monitor_performance(step, N_STEPS, simulation_start, Re)
            
            # Garbage collection every 100 steps
            if step % 100 == 0
                GC.gc()
            end
        end
        
        total_time = time() - simulation_start
        results[Re]["converged_steps"] = converged_steps
        results[Re]["total_time"] = total_time
        results[Re]["avg_step_time"] = total_time / N_STEPS
        
        println("\nRe = $Re completed:")
        println("  Converged steps: $(converged_steps)/$(N_STEPS)")
        println("  Total time: $(@sprintf("%.2f", total_time)) seconds")
        println("  Performance: $(@sprintf("%.2f", converged_steps/total_time)) steps/second")
        
        # Clean up memory
        GC.gc()
        sleep(0.5)
    end
    
    # Final summary
    println("\n" * "="^60)
    println("SIMULATION COMPLETED!")
    println("="^60)
    
    total_time = sum(results[Re]["total_time"] for Re in RE_VALUES if haskey(results[Re], "total_time"))
    total_steps = sum(results[Re]["converged_steps"] for Re in RE_VALUES if haskey(results[Re], "converged_steps"))
    
    println("Overall Performance:")
    println("  Total time: $(@sprintf("%.1f", total_time/60)) minutes")
    println("  Total steps: $total_steps")
    println("  Average: $(@sprintf("%.2f", total_steps/total_time)) steps/second")
    println("  Results: $base_dir")
    
    println("\nPer-Reynolds Performance:")
    for Re in RE_VALUES
        if haskey(results[Re], "total_time") && results[Re]["total_time"] > 0
            r = results[Re]
            println("  Re = $Re: $(r["converged_steps"])/$(N_STEPS) steps, $(@sprintf("%.1f", r["total_time"]))s, $(@sprintf("%.1f", r["converged_steps"]/r["total_time"])) steps/s")
        end
    end
    
    println("\n" * "="^60)
    return results
end

# Run the simulation
try
    results = run_optimized_simulation()
catch e
    println("ERROR: Simulation failed: $e")
    rethrow(e)
end