include("TRIPSSQP.jl")
struct TRIPSSQPResult
    KKTStep_Id::Array
    KKTendStep_Id::Array
    IterStep_Id::Array
    ComplexityStep_Id::Array
    KKTStep_SR1::Array
    KKTendStep_SR1::Array
    IterStep_SR1::Array
    ComplexityStep_SR1::Array
    KKTStep_EstH::Array
    KKTendStep_EstH::Array
    IterStep_EstH::Array
    ComplexityStep_EstH::Array
    KKTStep_AveH::Array
    KKTendStep_AveH::Array
    IterStep_AveH::Array
    ComplexityStep_AveH::Array
end

## Implement Adaptive SQP for whole problem set
# Adap: parameters of adaptive algorithm
# Prob: problem name set

function TRIPSSQPMain(TRIPSSQP, Prob)
    verbose = TRIPSSQP.verbose
    Max_Iter = TRIPSSQP.MaxIter
    EPS_Res = TRIPSSQP.EPS_Res
    mu = TRIPSSQP.mu
    delta_k = TRIPSSQP.delta_k
    epsilon_k = TRIPSSQP.epsilon_k
    theta_k = TRIPSSQP.theta_k
    TotalRep = TRIPSSQP.Rep
    rho = TRIPSSQP.rho
    gamma = TRIPSSQP.gamma
    delta_max = TRIPSSQP.delta_max
    kap_h = TRIPSSQP.kap_h
    kap_g = TRIPSSQP.kap_g
    kap_f = TRIPSSQP.kap_f
    epsilon_s = TRIPSSQP.epsilon_s
    xi = TRIPSSQP.xi
    p_h = TRIPSSQP.p_h
    p_g = TRIPSSQP.p_g
    p_f = TRIPSSQP.p_f
    eta = TRIPSSQP.eta
    C_grad = TRIPSSQP.C_grad
    Sigma = TRIPSSQP.Sigma
    LenC_grad = length(C_grad)
    LenSigma = length(Sigma)

    TRIPSSQPR = Array{TRIPSSQPResult}(undef,length(Prob))

    ## Go over all Problems
    for Idprob = 1:length(Prob)
        # load problem
        nlp = CUTEstModel(Prob[Idprob])
        seq = 1
#nlp = CUTEstModel("BT1")

        ## Identity

        # define results vectors
        KKTStep_Id = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKT1Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKT2Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        KKTendStep_Id = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKTend1Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKTend2Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        IterStep_Id = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        ComplexityStep_Id = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        # go over all cases
        i = 1
        while i <= LenC_grad
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRIPSSQP-Id","-",Idprob,"-",i,"-",j,"-",rep)
                    Ratio,Iter,Complexity = TRIPSSQPSQP(nlp,Sigma[j],Max_Iter,EPS_Res,mu,eta,gamma,rho,delta_k,delta_max,epsilon_k,epsilon_s,xi,kap_f,kap_g,kap_h,p_f,p_g,p_h,C_grad[i],1,theta_k,seq)
                    println("TRIPSSQP-Id","-",Idprob,"-",i,"-",j,"-",rep," ",Ratio[end]," ",Iter," ",Complexity)

                    push!(KKTStep_Id[i,j], Ratio)
                    push!(KKTendStep_Id[i,j], Ratio[end])
                    push!(IterStep_Id[i,j], Iter)
                    push!(ComplexityStep_Id[i,j], Complexity)
                    rep += 1

                end
                j += 1
            end
            i += 1
        end

        
        ## SR1
        # define results vectors
        KKTStep_SR1 = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKT1Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKT2Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        KKTendStep_SR1 = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKTend1Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKTend2Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        IterStep_SR1 = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        ComplexityStep_SR1 = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        # go over all cases
        i = 1
        while i <= LenC_grad
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRIPSSQP-SR1","-",Idprob,"-",i,"-",j,"-",rep)
                    Ratio,Iter,Complexity = TRIPSSQPSQP(nlp,Sigma[j],Max_Iter,EPS_Res,mu,eta,gamma,rho,delta_k,delta_max,epsilon_k,epsilon_s,xi,kap_f,kap_g,kap_h,p_f,p_g,p_h,C_grad[i],2,theta_k,seq)
                    println("TRIPSSQP-SR1","-",Idprob,"-",i,"-",j,"-",rep," ", Ratio[end]," ",Iter," ",Complexity)

                    push!(KKTStep_SR1[i,j], Ratio)
                    push!(KKTendStep_SR1[i,j], Ratio[end])
                    push!(IterStep_SR1[i,j], Iter)
                    push!(ComplexityStep_SR1[i,j], Complexity)
                    rep += 1

                end
                j += 1
            end
            i += 1
        end


        ## Estimated Hessian
        # define results vectors
        KKTStep_EstH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKT1Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKT2Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        KKTendStep_EstH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKTend1Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKTend2Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        IterStep_EstH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        ComplexityStep_EstH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        # go over all cases
        i = 1
        while i <= LenC_grad
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRIPSSQP-EstH","-",Idprob,"-",i,"-",j,"-",rep)
                    Ratio,Iter,Complexity = TRIPSSQPSQP(nlp,Sigma[j],Max_Iter,EPS_Res,mu,eta,gamma,rho,delta_k,delta_max,epsilon_k,epsilon_s,xi,kap_f,kap_g,kap_h,p_f,p_g,p_h,C_grad[i],3,theta_k,seq)
                    println("TRIPSSQP-EstH","-",Idprob,"-",i,"-",j,"-",rep," ", Ratio[end]," ",Iter," ",Complexity)

                    push!(KKTStep_EstH[i,j], Ratio)
                    push!(KKTendStep_EstH[i,j], Ratio[end])
                    push!(IterStep_EstH[i,j], Iter)
                    push!(ComplexityStep_EstH[i,j], Complexity)
                    rep += 1

                end
                j += 1
            end
            i += 1
        end



        ## Averaged Hessian
        # define results vectors
        KKTStep_AveH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKT1Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKT2Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        # EigenStep_AveH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        KKTendStep_AveH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKTend1Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        #KKTend2Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        # EigenendStep_AveH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        IterStep_AveH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        ComplexityStep_AveH = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        # go over all cases
        i = 1
        while i <= LenC_grad
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("TRIPSSQP-AveH","-",Idprob,"-",i,"-",j,"-",rep)
                    Ratio,Iter,Complexity = TRIPSSQPSQP(nlp,Sigma[j],Max_Iter,EPS_Res,mu,eta,gamma,rho,delta_k,delta_max,epsilon_k,epsilon_s,xi,kap_f,kap_g,kap_h,p_f,p_g,p_h,C_grad[i],4,theta_k,seq)
                    println("TRIPSSQP-AveH","-",Idprob,"-",i,"-",j,"-",rep," ", Ratio[end], " ",Iter," ",Complexity)

                    push!(KKTStep_AveH[i,j], Ratio)
                    push!(KKTendStep_AveH[i,j], Ratio[end])
                    push!(IterStep_AveH[i,j], Iter)
                    push!(ComplexityStep_AveH[i,j], Complexity)
                    rep += 1

                end
                j += 1
            end
            i += 1
        end



        #        TRIPSSQPR[Idprob] = TRIPSSQPResult(KKTStep,KKT1Step,KKT2Step,EigenStep,KKTendStep,KKTend1Step,KKTend2Step,EigenendStep,IterStep,KKTStepC,KKT1StepC,KKT2StepC,EigenStepC,KKTendStepC,KKTend1StepC,KKTend2StepC,EigenendStepC,IterStepC)
        TRIPSSQPR[Idprob] = TRIPSSQPResult(KKTStep_Id,KKTendStep_Id,IterStep_Id,ComplexityStep_Id,KKTStep_SR1,KKTendStep_SR1,IterStep_SR1,ComplexityStep_SR1,KKTStep_EstH,KKTendStep_EstH,IterStep_EstH,ComplexityStep_EstH,KKTStep_AveH,KKTendStep_AveH,IterStep_AveH,ComplexityStep_AveH)
        finalize(nlp)
    end
    return TRIPSSQPR
end
