module testpackage
using AxisArrays
using MCMCChains
using Optim
using LineSearches
using Random
using LinearAlgebra
using NaNMath
using LogExpFunctions
using PreallocationTools
using Distributions
using Turing
using Random
using ForwardDiff
using Logging
using CSV
using DataFrames

export optimize_many_MAP
export ChainsCustomIndex
export Chains 
export generate_pp_and_gq_eirrc
export fit_eirrc_closed
export NegativeBinomial2
export GeneralizedTDist
export power 
export new_eirrc_closed_solution!
export bayes_eirrc_closed!
# Write your package code here.
include("optimize_many_MAP.jl")
include("helper_functions.jl")
include("generate_pp_and_gq_eirrc.jl")
include("fit_eirrc_closed.jl")
include("distribution_functions.jl")
include("closed_soln_eirr_withincid.jl")
include("bayes_eirrc_closed.jl")
end
