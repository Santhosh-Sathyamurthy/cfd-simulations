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

# Parameters
L = 1.0  # Cylinder diameter
R_cylinder = L/2  # Cylinder radius
V_inf = 1.0  # Incoming velocity
domain_size = (20*L, 10*L)  # Domain size (length x width)
cylinder_center = (5*L, 0.0)  # Cylinder center

# Create mesh using gmsh geometry
function create_cylinder_mesh(h_far=0.5, h_near=0.05)
    # Generate gmsh geometry file
    geo_content = """
    // Geometry parameters
    L = $(L);
    R = $(R_cylinder);
    V_inf = $(V_inf);
    
    // Domain boundaries
    x_min = -$(5*L);
    x_max = $(15*L);
    y_min = -$(5*L);
    y_max = $(5*L);
    
    // Cylinder center
    cx = $(cylinder_center[1]);
    cy = $(cylinder_center[2]);
    
    // Characteristic lengths
    h_far = $(h_far);
    h_near = $(h_near);
    
    // Domain corners
    Point(1) = {x_min, y_min, 0, h_far};
    Point(2) = {x_max, y_min, 0, h_far};
    Point(3) = {x_max, y_max, 0, h_far};
    Point(4) = {x_min, y_max, 0, h_far};
    
    // Cylinder boundary points
    Point(5) = {cx, cy, 0, h_near};
    Point(6) = {cx + R, cy, 0, h_near};
    Point(7) = {cx, cy + R, 0, h_near};
    Point(8) = {cx - R, cy, 0, h_near};
    Point(9) = {cx, cy - R, 0, h_near};
    
    // Domain boundary lines
    Line(1) = {1, 2};
    Line(2) = {2, 3};
    Line(3) = {3, 4};
    Line(4) = {4, 1};
    
    // Cylinder arcs
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
    """
    
    # Write geometry file
    open("cylinder.geo", "w") do file
        write(file, geo_content)
    end
    
    # Generate mesh
    run(`gmsh -2 cylinder.geo -o cylinder.msh`)
    
    # Load mesh into Gridap
    model = GmshDiscreteModel("cylinder.msh")
    return model
end

# Create the mesh
println("Creating mesh...")
model = create_cylinder_mesh(0.3, 0.02)

# Define function spaces
order = 2
reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
reffe_p = ReferenceFE(lagrangian, Float64, order-1)

V = TestFESpace(model, reffe_u, conformity=:H1, 
                dirichlet_tags=["inlet", "walls", "cylinder"])
Q = TestFESpace(model, reffe_p, conformity=:H1)

# Boundary conditions
u_inlet = VectorValue(V_inf, 0.0)
u_walls = VectorValue(0.0, 0.0)
u_cylinder = VectorValue(0.0, 0.0)

U = TrialFESpace(V, [u_inlet, u_walls, u_cylinder])
P = TrialFESpace(Q)

# Multi-field spaces
Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

# Integration measures
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# Fixed Navier-Stokes solver
function solve_navier_stokes(ν, dt, u_prev, p_prev)
    println("    Solving Navier-Stokes...")
    
    # Create a NonlinearFEOperator for the implicit time-stepping
    function navier_stokes_residual(x, y)
        u, p = x
        v, q = y
        
        # Time derivative
        time_term = (u - u_prev) ⊙ v / dt
        
        # Viscous term
        viscous_term = ν * (∇(u) ⊙ ∇(v))
        
        # Convective term (linearized using previous velocity for stability)
        convective_term = (u_prev ⋅ ∇(u)) ⊙ v
        
        # Pressure terms
        pressure_term = - (∇ ⋅ v) * p + q * (∇ ⋅ u)
        
        return ∫(time_term + viscous_term + convective_term + pressure_term) * dΩ
    end
    
    # Create nonlinear operator
    op = FEOperator(navier_stokes_residual, X, Y)
    
    try
        # Solve the nonlinear problem (no initial guess needed)
        xh = solve(op)
        uh, ph = xh
        println("    Converged successfully")
        return uh, ph
    catch e
        println("    Warning: Solver failed with error: $e")
        println("    Using previous solution")
        return u_prev, p_prev
    end
end

# Alternative steady-state solver for initialization
function solve_steady_stokes(ν)
    println("    Solving steady Stokes for initialization...")
    
    # Stokes problem (no convection, no time derivative)
    function stokes_residual(x, y)
        u, p = x
        v, q = y
        
        # Only viscous and pressure terms
        viscous_term = ν * (∇(u) ⊙ ∇(v))
        pressure_term = - (∇ ⋅ v) * p + q * (∇ ⋅ u)
        
        return ∫(viscous_term + pressure_term) * dΩ
    end
    
    # Create linear operator
    op = FEOperator(stokes_residual, X, Y)
    
    try
        xh = solve(op)
        uh, ph = xh
        println("    Steady Stokes converged successfully")
        return uh, ph
    catch e
        println("    Warning: Steady Stokes failed: $e")
        # Fallback to simple initialization
        u0 = interpolate_everywhere(VectorValue(V_inf, 0.0), U)
        p0 = interpolate_everywhere(0.0, P)
        return u0, p0
    end
end

# Simplified vorticity calculation
function compute_vorticity(uh)
    # For 2D flow: ω = ∂v/∂x - ∂u/∂y
    function vorticity_at_point(x, y)
        try
            # Check if point is in domain and not too close to boundaries
            if (x - cylinder_center[1])^2 + (y - cylinder_center[2])^2 <= (R_cylinder + 0.01)^2
                return 0.0  # Inside or too close to cylinder
            end
            
            # Get velocity at point
            u_vec = uh(Point(x, y))
            
            # Use automatic differentiation for gradients if available
            # For now, use simple finite differences
            h = 1e-4
            
            try
                u_right = uh(Point(x + h, y))
                u_up = uh(Point(x, y + h))
                
                # Compute derivatives numerically
                dudy = (u_up[1] - u_vec[1]) / h
                dvdx = (u_right[2] - u_vec[2]) / h
                
                return dvdx - dudy
            catch
                return 0.0
            end
        catch e
            return 0.0
        end
    end
    
    return vorticity_at_point
end

# Simulation parameters - start with smaller Re for stability
Re_values = [40.0]  # Start with very low Re for stability
T = 2.0  # Shorter simulation time for testing
dt = 0.05  # Larger time step for faster testing
n_steps = Int(T / dt)

# Higher resolution plotting
plot_resolution = (800, 400)

for Re in Re_values
    ν = V_inf * L / Re
    println("Simulating Re = $Re with ν = $ν")
    
    # Initialize with steady Stokes solution for better convergence
    uh, ph = solve_steady_stokes(ν)
    
    # Store plots for animation
    plots_array = []
    
    # Time-stepping loop
    for step in 1:n_steps
        t = step * dt
        
        # Solve Navier-Stokes
        uh, ph = solve_navier_stokes(ν, dt, uh, ph)
        
        if step % 5 == 0  # Update every 5 steps for better performance
            println("  Step $step/$n_steps, t = $(round(t, digits=2))")
            
            # Compute vorticity
            vorticity_func = compute_vorticity(uh)
            
            # Create visualization points
            x_viz = range(-2*L, 12*L, length=60)  # Reduced resolution for speed
            y_viz = range(-3*L, 3*L, length=40)
            
            # Evaluate vorticity on visualization grid
            ω_vals = zeros(length(x_viz), length(y_viz))
            
            for (i, x) in enumerate(x_viz)
                for (j, y) in enumerate(y_viz)
                    # Check if point is inside cylinder
                    if (x - cylinder_center[1])^2 + (y - cylinder_center[2])^2 > R_cylinder^2
                        ω_vals[i,j] = vorticity_func(x, y)
                    else
                        ω_vals[i,j] = NaN
                    end
                end
            end
            
            # Plot vorticity field
            p1 = heatmap(x_viz, y_viz, ω_vals', 
                        aspect_ratio=:equal,
                        title="Vorticity Field, Re=$Re, t=$(round(t, digits=2))",
                        xlabel="x/L", ylabel="y/L",
                        colorbar_title="ω",
                        clim=(-2, 2),
                        size=plot_resolution,
                        dpi=100,
                        color=:RdBu)
            
            # Add cylinder outline
            θ = range(0, 2π, length=100)
            cyl_x = cylinder_center[1] .+ R_cylinder * cos.(θ)
            cyl_y = cylinder_center[2] .+ R_cylinder * sin.(θ)
            plot!(p1, cyl_x, cyl_y, color=:black, linewidth=2, label="Cylinder")
            
            # Create velocity magnitude plot
            u_vals = zeros(length(x_viz), length(y_viz))
            for (i, x) in enumerate(x_viz)
                for (j, y) in enumerate(y_viz)
                    if (x - cylinder_center[1])^2 + (y - cylinder_center[2])^2 > R_cylinder^2
                        try
                            u_vec = uh(Point(x, y))
                            u_vals[i,j] = sqrt(u_vec[1]^2 + u_vec[2]^2)
                        catch
                            u_vals[i,j] = 0.0
                        end
                    else
                        u_vals[i,j] = NaN
                    end
                end
            end
            
            p2 = heatmap(x_viz, y_viz, u_vals', 
                        aspect_ratio=:equal,
                        title="Velocity Magnitude, Re=$Re, t=$(round(t, digits=2))",
                        xlabel="x/L", ylabel="y/L",
                        colorbar_title="|u|",
                        clim=(0, 1.5),
                        size=plot_resolution,
                        dpi=100,
                        color=:viridis)
            
            plot!(p2, cyl_x, cyl_y, color=:black, linewidth=2, label="Cylinder")
            
            # Combined plot
            combined_plot = plot(p1, p2, layout=(2,1), size=(800, 800))
            push!(plots_array, combined_plot)
        end
    end
    
    # Create animation from stored plots
    if length(plots_array) > 0
        println("Creating animation for Re = $Re...")
        anim = @animate for i in 1:length(plots_array)
            plots_array[i]
        end
        
        # Save gif
        gif_filename = "cylinder_flow_Re$(Re).gif"
        gif(anim, gif_filename, fps=5)
        println("Saved: $gif_filename")
    end
    
    # Clear plots array to free memory
    plots_array = nothing
    GC.gc()  # Force garbage collection
end

println("Simulation complete!")