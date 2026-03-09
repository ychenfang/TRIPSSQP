# using Pkg
# Pkg.add("NLPModels")
# Pkg.add("JuMP")
# Pkg.add("LinearOperators")
# Pkg.add("OptimizationProblems")
# Pkg.add("MathProgBase")
# Pkg.add("ForwardDiff")
# Pkg.add("CUTEst")
# Pkg.add("NLPModelsJuMP")
# Pkg.add("LinearAlgebra")
# Pkg.add("Distributed")
# Pkg.add("Ipopt")
# Pkg.add("DataFrames")
# Pkg.add("PyPlot")
# Pkg.add("Glob")
# Pkg.add("DelimitedFiles")
# Pkg.add("Random")
# Pkg.add("Distributions")
# Pkg.add("MATLAB")
#Pkg.add("LinearAlgebra.LAPACK")
## Load packages
using NLPModels
using JuMP
using LinearOperators
using OptimizationProblems
using MathProgBase
using ForwardDiff
using CUTEst
using NLPModelsJuMP
using LinearAlgebra
using Distributed
using Ipopt
using DataFrames
using PyPlot
using MATLAB
using Glob
using DelimitedFiles
using Random
using Distributions
using LinearAlgebra.LAPACK

# cd("//Users/ycfang/Documents/Research/Opmization/StoSQP/NumericalAnalysis/2TR-SQP-STORM")

##
# define parameter module
module Parameter
    # Parameters of adaptive LS-SQP with l2 merit function

    struct Params
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Res::Float64                   # minimum of difference
        # adaptive parameters
        mu::Float64                        # mu
        delta_k::Float64                   # delta
        epsilon_k::Float64                 # epsilon
        theta_k::Float64                 #  barrier parameter
        # fixed parameters
        Rep::Int                           # Number of Independent runs
        rho::Float64                       # rho
        gamma::Float64                     # gamma
        delta_max::Float64                 # maximum of stepsize
        kap_h::Float64                     # kappa of Hessian
        kap_g::Float64                     # kappa of gradient
        kap_f::Float64                     # kappa of objective value
        epsilon_s::Float64                 # epsilon for slack
        xi::Float64                        # xi for controlling normal vs tangential
        p_h::Float64                       # prob of Hessian
        p_g::Float64                       # prob of gradient
        p_f::Float64                       # prob of f
        eta:: Float64                     # eta0
        # test parameters
        C_grad::Array{Float64}             # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end



end

################

##
include("TRIPSSQPMain.jl")
include("Param.jl")
using .Parameter
Prob = readdlm(string(pwd(),"problems.txt"))
#######################################
#########  run main file    ###########
#######################################
function main()
    Random.seed!(2028)
    
 
    TRIPSSQP = TRIPSSQPMain(TRIPSSQP, Prob)
    path = string(pwd(),"TRIPSSQP.mat")
    write_matfile(path; TRIPSSQP)
end

main()
