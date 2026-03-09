function sr1(B,bnab_xL,X)
    yk = bnab_xL[end] - bnab_xL[end-1]
    sk = X[end]-X[end-1]
    Bk = B[end]
    quant1 = yk- Bk*sk
    quant2 = quant1'*sk
    if norm(yk) ==0 || norm(sk)==0
        return Bk
    elseif quant2 < 1e-8*norm(sk)*norm(quant1)
        return Bk
    else
        B_new = Bk+quant1*quant1'/quant2
        if norm(B_new)>=100
            B_new=B[1]
        end
        return B_new
    end
end
