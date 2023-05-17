module testpackage
using Turing
using AxisArrays
using MCMCChains
using Optim
using LineSearches
using Random
using DrWatson
using LinearAlgebra
using ForwardDiff
using NaNMath
using Turing
using LogExpFunctions
using ForwardDiff
using PreallocationTools
using Distributions
using Turing
using Random
using ForwardDiff
using Logging
using PreallocationTools
using CSV
using DataFrames
using Random
using PreallocationTools

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
