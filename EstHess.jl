## This function estimate the gradient and Hessian for AdapSQP

function EstHess(nab_x2L_k, delta_k, sigma, nx, kap_h, p_grad, C_grad,Idd)
        if Idd == 1
                p_grad=p_grad/2
                Quant1 = delta_k^2*kap_h^2
                Quant2 = C_grad*log(1/(1-p_grad))/Quant1
                Xih = min(Quant2,1e5)
                Delta = rand(Normal(0,(sigma/Xih)^(1/2)), nx, nx)
                bnab_x2L_k = nab_x2L_k + Delta'*Delta/2
                return bnab_x2L_k, Xih
        else
                Delta = rand(Normal(0,(sigma)^(1/2)), nx, nx)
                bnab_x2L_k = nab_x2L_k + Delta'*Delta/2
                Xih = 1
                return bnab_x2L_k, Xih
        end
end
