# concRt

[![Build Status](https://github.com/igoldsteinh/testpackage.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/igoldsteinh/concRt.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package allows users to generate posterior estimates of the effective reproduction number from RNA concentrations and case counts and allows the user to fit the four models described in "Semiparametric Inference of Effective Reproduction Number
Dynamics from Wastewater Pathogen RNA Concentrations." 

## Required Installation Steps
To install the package type `]` to activate `pkg`, then use the command 
```
add https://github.com/igoldsteinh/concRt.jl
```

## Example usage
```
using concRt

data = [7.746720258212705, 8.480719432981955, 8.326951744058404, 7.7084576380183, 9.522542336229806, 8.22919522225484, 7.906042278483086, 8.016161975227881, 8.055971726781609, 8.819137047777467, 8.179845943869427, 9.279987578665471, 10.493308139257781, 10.498784254098249]
obstimes = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0]
param_change_times = [7.0, 14.0, 21.0]
priors_only = false
n_samples::Int64 = 10
n_chains::Int64 = 1

posterior_samples_eirr <- fit_eirrc(data, 
                               obstimes, 
                               param_change_times, 
                               priors_only, 
                               n_samples, 
                               n_chains, 
                               seed)
```
