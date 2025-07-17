# MIT License
# Copyright (c) 2025 Santhosh S
# See LICENSE file for full license text.

using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Fields
using GridapGmsh
using Plots
using LinearAlgebra
using SparseArrays
using ProgressMeter
using Printf
using Statistics
using Distributed
using SharedArrays

# Add processes for parallelization
if nprocs() == 1
    addprocs(min(8, Sys.CPU_THREADS - 4))  # Use 8 workers
end

@everywhere using Gridap
@everywhere using Plots

# Optimized simulation parameters
const L = 1.0
const R_cylinder = L/2
const V_inf = 1.0
const cylinder_center = (3*L, 0.0)  # Closer to inlet for efficiency


const T_TOTAL = 20.0  # Reduced total time
const DT = 0.05      # Larger time step for efficiency
const N_STEPS = Int(T_TOTAL / DT)
const SAVE_INTERVAL = 1  # Save every step

# Increasing the Stiffness Matrix Size
const H_FAR = 0.2
const H_NEAR = 0.02

# Convection-dominated Flow
const RE_VALUES = [100]

# Create output directories
function setup_output_directories()
    base_dir = "cylinder_flow_output"
    isdir(base_dir) || mkdir(base_dir)
    
    for Re in RE_VALUES
        re_dir = joinpath(base_dir, "Re_$(Re)")
        isdir(re_dir) || mkdir(re_dir)
    end
    
    return base_dir
end

# Much more efficient mesh
function create_efficient_mesh(h_far=H_FAR, h_near=H_NEAR)
    geo_content = """
    // Efficient geometry
    L = $(L);
    R = $(R_cylinder);
    
    // Smaller domain for efficiency
    x_min = -$(L);
    x_max = $(8*L);
    y_min = -$(2*L);
    y_max = $(2*L);
    
    // Cylinder center
    cx = $(cylinder_center[1]);
    cy = $(cylinder_center[2]);
    
    // Mesh sizes
    h_far = $(h_far);
    h_near = $(h_near);
    
    // Domain corners
    Point(1) = {x_min, y_min, 0, h_far};
    Point(2) = {x_max, y_min, 0, h_far};
    Point(3) = {x_max, y_max, 0, h_far};
    Point(4) = {x_min, y_max, 0, h_far};
    
    // Cylinder
    Point(5) = {cx, cy, 0, h_near};
    Point(6) = {cx + R, cy, 0, h_near};
    Point(7) = {cx, cy + R, 0, h_near};
    Point(8) = {cx - R, cy, 0, h_near};
    Point(9) = {cx, cy - R, 0, h_near};
    
    // Domain lines
    Line(1) = {1, 2};
    Line(2) = {2, 3};
    Line(3) = {3, 4};
    Line(4) = {4, 1};
    
    // Cylinder
    Circle(5) = {6, 5, 7};
    Circle(6) = {7, 5, 8};
    Circle(7) = {8, 5, 9};
    Circle(8) = {9, 5, 6};
    
    // Line loops
    Line Loop(1) = {1, 2, 3, 4};
    Line Loop(2) = {5, 6, 7, 8};
    
    // Surface
    Plane Surface(1) = {1, 2};
    
    // Physical entities
    Physical Line("inlet") = {4};
    Physical Line("outlet") = {2};
    Physical Line("walls") = {1, 3};
    Physical Line("cylinder") = {5, 6, 7, 8};
    Physical Surface("domain") = {1};
    
    // Efficient mesh settings
    Mesh.Algorithm = 6;
    Mesh.CharacteristicLengthMin = $(h_near);
    Mesh.CharacteristicLengthMax = $(h_far);
    """
    
    open("cylinder_efficient.geo", "w") do file
        write(file, geo_content)
    end
    
    run(`gmsh -2 cylinder_efficient.geo -o cylinder_efficient.msh`)
    model = GmshDiscreteModel("cylinder_efficient.msh")
    return model
end

# Efficient solver with lower-order elements
function create_efficient_solver(model, Re)
    ν = V_inf * L / Re
    
    # Use P1-P1 elements with stabilization for efficiency
    order = 1
    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    reffe_p = ReferenceFE(lagrangian, Float64, order)
    
    # Test spaces
    V = TestFESpace(model, reffe_u, conformity=:H1, 
                    dirichlet_tags=["inlet", "walls", "cylinder"])
    Q = TestFESpace(model, reffe_p, conformity=:H1)
    
    # Boundary conditions
    u_inlet = VectorValue(V_inf, 0.0)
    u_walls = VectorValue(0.0, 0.0)
    u_cylinder = VectorValue(0.0, 0.0)
    
    # Trial spaces
    U = TrialFESpace(V, [u_inlet, u_walls, u_cylinder])
    P = TrialFESpace(Q)
    
    # Multi-field spaces
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    
    # Integration
    degree = 2*order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    
    # Stabilization parameter
    h = H_NEAR
    τ = h^2 / (4*ν + 2*V_inf*h)
    
    return X, Y, dΩ, ν, τ
end

# Fast Stokes solver
function solve_stokes_fast(X, Y, dΩ, ν, τ)
    function stokes_residual(x, y)
        u, p = x
        v, q = y
        
        # Standard Stokes with simple stabilization
        viscous = ν * (∇(u) ⊙ ∇(v))
        pressure = - (∇ ⋅ v) * p + q * (∇ ⋅ u)
        stabilization = τ * (∇(p) ⊙ ∇(q))
        
        return ∫(viscous + pressure + stabilization) * dΩ
    end
    
    op = FEOperator(stokes_residual, X, Y)
    
    try
        xh = solve(op)
        uh, ph = xh
        return uh, ph, true
    catch e
        println("    Stokes failed: $e")
        return nothing, nothing, false
    end
end

# Fast Navier-Stokes step
function solve_ns_fast(X, Y, dΩ, ν, τ, dt, u_prev, p_prev)
    function ns_residual(x, y)
        u, p = x
        v, q = y
        
        # Efficient semi-implicit scheme
        time_term = (u - u_prev) ⊙ v / dt
        viscous_term = ν * (∇(u) ⊙ ∇(v))
        convective_term = ((u_prev ⋅ ∇(u)) ⊙ v)
        pressure_term = - (∇ ⋅ v) * p + q * (∇ ⋅ u)
        stabilization = τ * (∇(p) ⊙ ∇(q))
        
        return ∫(time_term + viscous_term + convective_term + pressure_term + stabilization) * dΩ
    end
    
    op = FEOperator(ns_residual, X, Y)
    
    try
        xh = solve(op)
        uh, ph = xh
        return uh, ph, true
    catch e
        return u_prev, p_prev, false
    end
end

# Much faster visualization with parallel processing
@everywhere function compute_field_chunk(x_chunk, y_range, uh, cylinder_center, R_cylinder)
    nx, ny = length(x_chunk), length(y_range)
    vorticity = zeros(nx, ny)
    velocity_mag = zeros(nx, ny)
    
    for (i, x) in enumerate(x_chunk)
        for (j, y) in enumerate(y_range)
            # Skip cylinder interior
            if (x - cylinder_center[1])^2 + (y - cylinder_center[2])^2 > (R_cylinder * 1.1)^2
                try
                    point = Point(x, y)
                    u_val = uh(point)
                    
                    # Velocity magnitude
                    velocity_mag[i, j] = sqrt(u_val[1]^2 + u_val[2]^2)
                    
                    # Simple vorticity estimate
                    h = 0.05
                    if i > 1 && i < nx && j > 1 && j < ny
                        try
                            u_x = uh(Point(x + h, y))
                            u_y = uh(Point(x, y + h))
                            
                            dudy = (u_y[1] - u_val[1]) / h
                            dvdx = (u_x[2] - u_val[2]) / h
                            
                            vorticity[i, j] = dvdx - dudy
                        catch
                            vorticity[i, j] = 0.0
                        end
                    end
                catch
                    vorticity[i, j] = 0.0
                    velocity_mag[i, j] = 0.0
                end
            else
                vorticity[i, j] = NaN
                velocity_mag[i, j] = NaN
            end
        end
    end
    
    return vorticity, velocity_mag
end

# Efficient visualization with better text positioning
function create_efficient_visualization(uh, ph, Re, step, output_dir)
    try
        # Smaller, more efficient grid
        x_range = range(-0.5*L, 7*L, length=300)
        y_range = range(-1.5*L, 1.5*L, length=200)
        
        # Split x_range for parallel processing
        n_chunks = min(nprocs(), 4)
        x_chunks = [x_range[i:n_chunks:end] for i in 1:n_chunks]
        
        # Process chunks in parallel
        results = pmap(chunk -> compute_field_chunk(chunk, y_range, uh, cylinder_center, R_cylinder), x_chunks)
        
        # Reconstruct full arrays
        vorticity = zeros(length(x_range), length(y_range))
        velocity_mag = zeros(length(x_range), length(y_range))
        
        for (i, (vort_chunk, vel_chunk)) in enumerate(results)
            indices = i:n_chunks:length(x_range)
            vorticity[indices, :] = vort_chunk
            velocity_mag[indices, :] = vel_chunk
        end
        
        # Scale vorticity
        valid_vort = filter(!isnan, vec(vorticity))
        if length(valid_vort) > 0
            vort_scale = quantile(abs.(valid_vort), 0.9)
            if vort_scale > 0
                vorticity = clamp.(vorticity ./ vort_scale, -2, 2)
            end
        end
        
        # Cylinder outline
        θ = range(0, 2π, length=50)
        cyl_x = cylinder_center[1] .+ R_cylinder * cos.(θ)
        cyl_y = cylinder_center[2] .+ R_cylinder * sin.(θ)
        
        # Create efficient plots
        theme(:dark)
        
        # Vorticity plot with better text positioning
        p1 = heatmap(x_range, y_range, vorticity',
                     c=:plasma,
                     aspect_ratio=:equal,
                     size=(800, 500),
                     dpi=100,  # Lower DPI for speed
                     clim=(-2, 2),
                     axis=false,
                     showaxis=false,
                     legend=:none,
                     ticks=false,
                     margin=0.0Plots.mm)
        
        plot!(p1, cyl_x, cyl_y, color=:white, linewidth=2, label="")
        
        # Better text positioning
        t_current = step * DT
        text_x = x_range[1] + 0.1 * (x_range[end] - x_range[1])
        text_y = y_range[end] - 0.1 * (y_range[end] - y_range[1])
        
        annotate!(p1, text_x, text_y, 
                  text("Re = $Re, t = $(@sprintf("%.1f", t_current))", 
                       pointsize=12, color=:white, halign=:left, valign=:top))
        
        # Save
        filename = joinpath(output_dir, "vorticity_$(@sprintf("%04d", step)).png")
        savefig(p1, filename)
        
        # Velocity plot
        p2 = heatmap(x_range, y_range, velocity_mag',
                     c=:viridis,
                     aspect_ratio=:equal,
                     size=(800, 500),
                     dpi=100,
                     clim=(0, 1.5),
                     axis=false,
                     showaxis=false,
                     legend=:none,
                     ticks=false,
                     margin=0.0Plots.mm)
        
        plot!(p2, cyl_x, cyl_y, color=:white, linewidth=2, label="")
        
        annotate!(p2, text_x, text_y, 
                  text("Velocity, Re = $Re", 
                       pointsize=12, color=:white, halign=:left, valign=:top))
        
        filename_vel = joinpath(output_dir, "velocity_$(@sprintf("%04d", step)).png")
        savefig(p2, filename_vel)
        
        return true
    catch e
        println("    Visualization failed: $e")
        return false
    end
end

# Efficient main simulation
function run_efficient_simulation()
    println("Starting efficient cylinder flow simulation...")
    println("Using $(nprocs()) processes for parallel computation")
    
    # Setup
    base_dir = setup_output_directories()
    
    # Create efficient mesh
    println("Creating efficient mesh...")
    model = create_efficient_mesh()
    println("Mesh created with fewer elements for speed")
    
    # Process each Reynolds number
    for Re in RE_VALUES
        println("\n" * "="^50)
        println("Processing Reynolds number: $Re")
        println("="^50)
        
        output_dir = joinpath(base_dir, "Re_$(Re)")
        
        # Setup efficient solver
        println("Setting up efficient solver...")
        X, Y, dΩ, ν, τ = create_efficient_solver(model, Re)
        
        # Fast Stokes solution
        println("Solving Stokes (fast)...")
        uh, ph, stokes_ok = solve_stokes_fast(X, Y, dΩ, ν, τ)
        
        if !stokes_ok
            println("ERROR: Stokes solver failed for Re = $Re")
            continue
        end
        
        # Initial visualization
        create_efficient_visualization(uh, ph, Re, 0, output_dir)
        
        # Efficient time stepping
        println("Time integration (efficient)...")
        converged_steps = 0
        
        @showprogress "Re = $Re: " for step in 1:N_STEPS
            # Fast NS solve
            uh_new, ph_new, converged = solve_ns_fast(X, Y, dΩ, ν, τ, DT, uh, ph)
            
            if converged
                uh, ph = uh_new, ph_new
                converged_steps += 1
            end
            
            # Efficient visualization
            if step % SAVE_INTERVAL == 0
                create_efficient_visualization(uh, ph, Re, step, output_dir)
            end
        end
        
        println("Completed Re = $Re ($(converged_steps)/$(N_STEPS) converged)")
        
        # Cleanup
        GC.gc()
    end
    
    println("\n" * "="^50)
    println("Efficient simulation completed!")
    println("Results in: $base_dir")
    println("Total processes used: $(nprocs())")
    println("="^50)
end

# Run the efficient simulation
run_efficient_simulation()