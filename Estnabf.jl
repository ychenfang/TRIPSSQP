## This function estimate the gradient and Hessian for AdapSQP

function Estnabf(nabf_k, delta_k, CovM,kap_g,p_grad,C_grad)
        p_grad = p_grad/2
        Quant1 = min(delta_k^4,delta_k^2)*kap_g^2
        Quant2 = C_grad*log(1/(1-p_grad))/Quant1
        Xi = min(Quant2,1e5)
        bnabf_k = mean(rand(MvNormal(nabf_k, CovM), convert(Int64, floor(Xi))),dims = 2)
        bnabf_k = vec(bnabf_k)
        return bnabf_k, Xi
end
