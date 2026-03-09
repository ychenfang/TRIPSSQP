include("Estnabf.jl")
include("EstHess.jl")
include("Estf.jl")
include("ProjCG.jl")
include("sr1.jl")
# Arguments
# - `nlp`        : NLP problem (CUTEst or NLPModels interface)
# - `sigma`      : noise standard deviation for gradient/Hessian approximations
# - `Max_Iter`   : maximum number of iterations
# - `EPS_Res`    : convergence tolerance on relative KKT residual
# - `mu`         : initial penalty parameter for the merit function
# - `eta`        : sufficient decrease ratio threshold (e.g. 0.4)
# - `gamma`      : trust-region expansion/contraction factor (e.g. 1.3)
# - `rho`        : penalty parameter growth rate (e.g. 1.2)
# - `delta_k`    : initial trust-region radius
# - `delta_max`  : maximum trust-region radius
# - `epsilon_k`  : initial criticality measure tolerance
# - `epsilon_s`  : slack feasibility fraction (e.g. 0.99)
# - `xi`         : normal step scaling parameter (e.g. 0.6)
# - `kap_f`      : function estimation accuracy constant
# - `kap_g`      : gradient estimation accuracy constant
# - `kap_h`      : Hessian estimation accuracy constant
# - `p_f`        : function estimation exponent
# - `p_g`        : gradient estimation exponent
# - `p_h`        : Hessian estimation exponent
# - `C_grad`     : gradient model constant
# - `Id`         : Hessian approximation type
#                     1 = Identity
#                     2 = SR1 quasi-Newton
#                     3 = Estimated Hessian (noisy finite-difference)
#                     4 = Averaged estimated Hessian (window of 50)
# - `theta_k`    : initial log-barrier parameter
# - `Id2`        : barrier parameter update rule
#                     1 = slow decay  θ_k = θ₀ · k^{-0.1}
#                     2 = growth then decay (experimental)

# Returns
# - `Ratio`      : history of relative KKT residuals ‖KKT_k‖/max(‖KKT_1‖, 1)
# - `iter`       : total number of iterations performed
# - `Flops`      : approximate floating-point operation count
using SparseArrays
using Plots

function TRIPSSQPSQP(nlp,sigma,Max_Iter,EPS_Res,mu,eta,gamma,rho,delta_k,delta_max,epsilon_k,epsilon_s,xi,kap_f,kap_g,kap_h,p_f,p_g,p_h,C_grad,Id,theta_k,Id2)
    nx = nlp.meta.nvar
    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon
    lvar = nlp.meta.lvar
    uvar = nlp.meta.uvar

    theta = theta_k
    # number of equalities and inequalities
    eq_con_idx = findall(lcon .== ucon)
    ineq_lower_con_idx = findall(lcon .> -Inf .&& lcon .< ucon)
    ineq_upper_con_idx = findall(ucon .< Inf .&& lcon .< ucon)

    #eq_var_idx = findall(nlp.meta.lvar .== nlp.meta.uvar)
    ineq_lower_var_idx = findall(lvar .> -Inf .&& lvar .< uvar)
    ineq_upper_var_idx = findall(uvar .< Inf .&& lvar .< uvar)


    I_nx = spdiagm(nx, nx, 0 => ones(nx))

    # Indicator of convergence and singularity
    IdCon, IdSing = 1, 0
    # Initialize
    # number of slack variables
    ns = length(ineq_lower_con_idx) + length(ineq_upper_con_idx) + length(ineq_lower_var_idx) + length(ineq_upper_var_idx)

   
    

    # store Data
    bnab_x2L = []
    push!(bnab_x2L,Diagonal(ones(nx)))
    bnabf = []
    KKT = []
    Q = []
    KKT_norm = []  # just to start
    Q_norm = []
    W = []

    Ratio = []
    Flops = 0

    # Initialization
    iter =0
    X = [nlp.meta.x0] 
    S = [1*ones(ns)]
   

    
    Complexity = 0

    

    for k in 1:Max_Iter
        iter = k
        x_k = X[end]
        s_k = S[end]
        ## Obtain est of bnab_xf_k


        # evaluate constraint and Jacobian
        c_full, J_full = consjac(nlp, x_k)

        # Equality residual and Jacobian
        c_k = c_full[eq_con_idx]
        G_k = J_full[eq_con_idx, :]

        # Inequality residual (all converted to h(x,s) ≤ 0 form)
        h_k = vcat(
            -c_full[ineq_lower_con_idx] .+ lcon[ineq_lower_con_idx],
             c_full[ineq_upper_con_idx] .- ucon[ineq_upper_con_idx],
            -x_k[ineq_lower_var_idx]   .+ lvar[ineq_lower_var_idx],
             x_k[ineq_upper_var_idx]   .- uvar[ineq_upper_var_idx]
        )

        # Inequality Jacobian (rows correspond to h_k)
        J_k = vcat(
            -J_full[ineq_lower_con_idx, :],
             J_full[ineq_upper_con_idx, :],
            -I_nx[ineq_lower_var_idx, :],
             I_nx[ineq_upper_var_idx, :]
        )
        A_k = [G_k  spzeros(length(eq_con_idx), ns);
               J_k  Diagonal(s_k)]

        
        AAT = Matrix(A_k * A_k')
        P_k = try
            I - A_k' * (AAT \ Matrix(A_k))   # P = I - Aᵀ(AAᵀ)⁻¹A
        catch
            IdSing = 1
            nothing
        end

        if IdSing == 1
            @warn "Singular constraint Jacobian at iteration $k; aborting."
            return Ratio, k, Flops
        end
        Flops += size(AAT, 1)^2   # dense solve cost proxy

        # evaluate objective, gradient, Hessian
        f_k, nabf_k = objgrad(nlp, x_k)

        CovM = sigma * (I + ones(nx, nx))   # isotropic + rank-1 noise model
        bnabf_k, _ = Estnabf(nabf_k, delta_k, CovM, kap_g, p_g, C_grad)
        push!(bnabf_hist, bnabf_k)

        # ψ vectors: true and estimated (augmented with slack penalty)
        psi_k     = [nabf_k;   -theta_k * ones(ns)]
        barpsi_k  = [bnabf_k;  -theta_k * ones(ns)]
        vartheta_k = [c_k; h_k .+ s_k]

        # Criticality measure
        Q_k       = [P_k * psi_k;    vartheta_k]
        barQ_k    = [P_k * barpsi_k; vartheta_k]
        Qnorm_k   = norm(Q_k)
        barQnorm_k = norm(barQ_k)
        push!(Q_norm, Qnorm_k)
        
        
         #  True KKT multipliers (for diagnostics)
        # -------------------------------------------------------------- #
        tau_k = theta_k ./ s_k                        # slack multipliers
        GGT   = Matrix(G_k * G_k')
        lam_k = -(GGT \ (G_k * (nabf_k .+ J_k' * tau_k)))

        # KKT residual components
        KKT_k1 = nabf_k .+ G_k' * lam_k .+ J_k' * tau_k   # stationarity
        KKT_k2 = c_k                                         # equality feas.
        KKT_k3 = max.(h_k, 0)                               # ineq. feas.
        KKT_k4 = min.(tau_k, 0)                             # dual feas.
        KKT_k5 = min.(-h_k, tau_k)                          # complementarity
        KKT_k  = [KKT_k1; KKT_k2; KKT_k3; KKT_k4; KKT_k5]

        push!(KKT_norm,  norm(KKT_k))
        push!(KKT_norm1, norm(KKT_k1))
        push!(KKT_norm5, norm(KKT_k5))
        push!(Ratio, KKT_norm[end] / max(KKT_norm[1], 1.0))
         
       #  Lagrangian Hessian (true, for reference in Id == 3,4)
        #
        #  ∇²_xx L = ∇²f(x) + Σᵢ λᵢ ∇²cᵢ(x) + Σⱼ τⱼ ∇²hⱼ(x)
        #
        #  NLPModels provides  hess(nlp, x; obj_weight, y)  which returns
        #    obj_weight * ∇²f(x) + Σᵢ yᵢ ∇²cᵢ(x)
        #  so we build y = [lam_k; tau_ineq] properly.
        # -------------------------------------------------------------- #
        n_eq    = length(eq_con_idx)
        n_ineq  = ns   # one slack per inequality side

        # Multipliers for the full constraint vector expected by NLPModels:
        # sign conventions follow the h_k construction above.
        y_full = zeros(nlp.meta.ncon)
        y_full[eq_con_idx] .= lam_k

        # For inequality constraints (con bounds), reconstruct signed multipliers
        y_full[ineq_lower_con_idx] .-= tau_k[1:length(ineq_lower_con_idx)]
        y_full[ineq_upper_con_idx] .+= tau_k[
            length(ineq_lower_con_idx) .+ (1:length(ineq_upper_con_idx))
        ]
        # Variable-bound inequalities have no second-order constraint Hessian.

        # True Lagrangian Hessian (lower triangle, sparse)
        nab_x2L_k = hess(nlp, x_k; obj_weight = 1.0, y = y_full)
        
       


        
        # estimate Hessian
        if Id ==1 
            # Identity
            bnab_x2L_k = Diagonal(ones(nx))
            Flops = Flops + nx
        elseif Id == 2
            # SR1
            if k>=2
                bnab_x2L_k = sr1(bnab_x2L,bnabf,X)
                push!(bnab_x2L,bnab_x2L_k)
                Flops = Flops + nx^2
            else
                bnab_x2L_k = bnab_x2L[end]
                #push!(bnab_x2L,bnab_x2L_k)
                Flops = Flops + nx^2  
                
            end
        elseif Id == 3
            # Estimated Hessian
            bnab_x2L_k , xih = EstHess(nab_x2L_k, delta_k, sigma, nx, kap_h, p_h, C_grad,0)
            push!(bnab_x2L,bnab_x2L_k)
            Flops = Flops + nx^2  + size(Matrix(G_k*G_k'),1)^2

        elseif Id == 4
            # Averaged Hessian
            H_k , xih = EstHess(nab_x2L_k, delta_k, sigma, nx, kap_h, p_h, C_grad,0)
            push!(bnab_x2L,H_k)
            Flops = Flops + nx^2  + size(Matrix(G_k*G_k'),1)^2
            if k<=50
                bnab_x2L_k = mean(bnab_x2L[1:end-1])
            else
                bnab_x2L_k = mean(bnab_x2L[end-50:end-1])
            end
            ##averaging
        # elseif Id == 5
        #     bnab_x2L_k, xih = EstHess(nab_x2L_k, delta_k, sigma, nx, kap_h, p_h, C_grad,1)
        #     push!(bnab_x2L,bnab_x2L_k)
        end

        W_k = [bnab_x2L_k spzeros(nx, ns); spzeros(ns, nx) Diagonal(theta_k*ones(ns))]
        push!(W, W_k)




    
        # Check if the iteration is unsuccessful
        if barQnorm_k / max(1.0, norm(W_k)) < eta * delta_k
            push!(X, x_k)
            push!(S, s_k)
            delta_k   /= gamma
            epsilon_k /= gamma
            continue
        end
        
    
        # compute the normal step    
        v_k = -A_k' * (AAT \ vartheta_k)
        gamma_k = min(
            xi * epsilon_s / max(norm(v_k[nx+1:end]), 1e-14),
            xi * delta_k   / max(norm(v_k),           1e-14),
            1.0
        )
        w_k = gamma_k * v_k

    

        # compute the tangential step
       hatDelta_k = sqrt(max(delta_k^2 - norm(w_k)^2, 0.0))
        slack_lower_bd = -epsilon_s * ones(ns) .- w_k[nx+1:end]
        t_k = solve_trust_region_general(
            W_k, barpsi_k .+ gamma_k .* (W_k * v_k),
            A_k, hatDelta_k, ns, slack_lower_bd
        )
        Flops += (nx + ns)^2
        ## trial step deltax_k
        ## rescaled trial step
        tilde_d_k = w_k .+ t_k
        deltax_k  = tilde_d_k[1:nx]
        deltas_k  = Diagonal(s_k) * tilde_d_k[nx+1:end]
        

        
        # compute predicted reduction:
        # update mu
        lower_bd_pred_k = -1/2*barQnorm_k * min(delta_k, epsilon_s, barQnorm_k/norm(W_k))  #set kappa_fcd = 1

        pred_k = barpsi_k' * tilde_d_k + 1/2 * tilde_d_k' * W_k * tilde_d_k + mu*( norm( vartheta_k + A_k * tilde_d_k ) - norm( vartheta_k ) )
        while true
           pred_k = barpsi_k' * tilde_d_k + 1/2 * tilde_d_k' * W_k * tilde_d_k + mu*( norm( vartheta_k + A_k * tilde_d_k ) - norm( vartheta_k ) )
            if mu > 1e8
                mu = 1e8
                break
            elseif pred_k > lower_bd_pred_k
                mu *= rho
            else
                break
            end
        end
      

        # Estimate function value
        bL_mu_k, bL_mu_sk = Estf(nlp,sigma,f_k,kap_f,p_f,c_k,h_k,X[end],S[end],mu,deltax_k,deltas_k,delta_k,epsilon_k,theta_k, eq_con_idx, ineq_lower_con_idx, ineq_upper_con_idx, ineq_lower_var_idx, ineq_upper_var_idx,lcon,ucon,lvar,uvar)

        # if k>=3000 && mod(k,100)==0
        #     println("bL_mu_k=",bL_mu_k)
        #     println("bL_mu_sk=",bL_mu_sk)
        # end
        # compute actual reduction:
        ared_k = bL_mu_sk - bL_mu_k

        # if k>=3000 && mod(k,100)==0
        #     println("ared_k=", ared_k)
        # end
        # decide whether the trial step is successful:
        brho = ared_k/pred_k
        # if k>=3000 && mod(k,100)==0
        #     println("brho=", brho)
        # end

# delta_max = 5
# gamma = 1.3

        if brho >= eta
            push!(X, x_k .+ deltax_k)
            push!(S, s_k .+ deltas_k)
            delta_k   = min(delta_max, gamma * delta_k)
            epsilon_k = (-pred_k >= epsilon_k) ? gamma * epsilon_k : epsilon_k / gamma
        else
            push!(X, x_k)
            push!(S, s_k)
            delta_k   /= gamma
            epsilon_k /= gamma
        end
        
        if Id2 == 1
            theta_k = theta_k * 0.9999
        elseif Id2 == 2
            theta_k = theta * k^(-0.1)
        end
        
        if Ratio[end] < EPS_Res
            println(nlp,"converged at iteration $k")
            println("KKT=",KKT_norm[end])
            println("KKT1=",KKT_norm1[end])
            println("KKT5=",KKT_norm5[end])
            println("Relative KKT=",Ratio[end])
            break
        end

        if Flops >= 2*1e6
            println(nlp,"Flops used up at iteration $k with flops= $Flops")
            println("KKT=",KKT_norm[end])
            println("KKT1=",KKT_norm1[end])
            println("KKT5=",KKT_norm5[end])
            println("Relative KKT=",Ratio[end])
            break
        end
    end
    
    
    return Ratio, iter, Flops;
    
end

