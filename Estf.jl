## estimate the objective value
function Estf(nlp,sigma,f_k,kap_f,p_f,c_k,h_k,x,s,mu,deltax_k,deltas_k,delta_k,epsilon_k,theta_k,eq,ineq_lower,ineq_upper,ineq_lower_var,ineq_upper_var,lcon,ucon,lvar,uvar)
    C_grad = 5
    Quant_f = min(kap_f^2*delta_k^4,epsilon_k^2)

    p_f = p_f/2
    Xi_f = min(C_grad/((1-p_f)*Quant_f), 1e5)
    if isnan(Xi_f)
        Xi_f = 1e5
    end
    # estimate f
    bf_k = f_k+rand(Normal(0,(sigma/Xi_f)^(1/2)))
    bL_mu_k = bf_k+ mu*norm([c_k;h_k+s]) - theta_k* sum(log.(s))
    # estimate f_s_k
    x_sk = x+deltax_k
    s_sk = s+deltas_k
    f_sk, _ = objgrad(nlp,vec(x_sk))
    consjac_sk = consjac(nlp,vec(x_sk))

    c_sk = consjac_sk[1][eq]
    h_sk = vcat(-consjac_sk[1][ineq_lower]+lcon[ineq_lower], consjac_sk[1][ineq_upper]-ucon[ineq_upper], -x_sk[ineq_lower_var]+lvar[ineq_lower_var], x_sk[ineq_upper_var]-uvar[ineq_upper_var])
    
    bf_sk = f_sk + rand(Normal(0,(sigma/Xi_f)^(1/2)))
    bL_mu_sk = bf_sk+ mu*norm([c_sk;h_sk+s_sk]) - theta_k* sum(log.(s_sk))
    return bL_mu_k, bL_mu_sk
end
