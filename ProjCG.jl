using JuMP
using Ipopt 

"""
Solves the constrained trust-region subproblem using a general-purpose solver.

Arguments:
- Wk_tilde: The Hessian matrix (W_k)
- psik_tilde_plus_gamma_Wk_vk: The gradient vector (g)  
- Ak: The equality constraint matrix (Ak)
- Delta_tilde: The trust-region radius (Delta)
- n_last: The number of last entries for the new constraint
- v_lower_bound: The vector 'v' for the new constraint
"""
function solve_trust_region_general(
    Wk_tilde, 
    psik_tilde_plus_gamma_Wk_vk, 
    Ak, 
    Delta_tilde,
    n_last::Int,
    v_lower_bound::Vector
)
    
    d = size(Wk_tilde, 1) # Total dimension of t
    num_equalities = size(Ak, 1)

    if length(v_lower_bound) != n_last
        error("Dimension mismatch: v_lower_bound must have length n_last")
    end

    # 1. Create the JuMP model
    # We use Ipopt, a powerful solver for non-linear problems (including QCQPs)
    model = Model(Ipopt.Optimizer)
    set_silent(model) # Suppress solver output

    # 2. Define the variable 't'
    @variable(model, t[1:d])

    # 3. Define the objective function
    @objective(
        model, 
        Min, 
        0.5 * dot(t, Wk_tilde * t) + dot(psik_tilde_plus_gamma_Wk_vk, t)
    )

    # 4. Define the constraints
    
    # Equality constraint: Ak * t = 0
    if num_equalities > 0
        @constraint(model, eq_con, Ak * t .== zeros(num_equalities))
    end

    # Trust-region constraint: ||t||^2 <= Delta_tilde^2
    @constraint(model, tr_con, dot(t, t) <= Delta_tilde^2)

    # ---- YOUR NEW CONSTRAINT ----
    # Get the indices for the last 'n' entries
    last_n_indices = (d - n_last + 1):d
    
    @constraint(
        model, 
        ineq_con, 
        t[last_n_indices] .>= v_lower_bound
    )
    # -----------------------------

    # 5. Solve the problem
    optimize!(model)

    # 6. Return the solution
    if termination_status(model) == MOI.LOCALLY_SOLVED || termination_status(model) == MOI.OPTIMAL
        return value.(t)
    else
        @warn("Subproblem solver did not find an optimal solution. Status: $(termination_status(model))")
        # Return a feasible point (e.g., zeros) or handle the error
        return zeros(d) 
    end
end

# --- How you would call it ---
# H = ...
# g = ...
# G = ...
# Delta = 1.0
# n = 2
# v = [-0.1, -0.1]

# t_solution = solve_trust_region_general(H, g, G, Delta, n, v)