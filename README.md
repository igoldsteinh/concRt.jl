# testpackage

[![Build Status](https://github.com/igoldsteinh/testpackage.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/igoldsteinh/testpackage.jl/actions/workflows/CI.yml?query=branch%3Amain)

# testpackage
This package allows users to generate posterior estimates of the effective reproduction number from RNA concentrations and case counts and allows the user to fit the four models described in ``Semiparametric Inference of Effective Reproduction Number
Dynamics from Wastewater Pathogen RNA Concentrations." 

## Required Installation Steps
To install the package type `]` to activate `pkg`, then use the command 
```
add https://github.com/igoldsteinh/testpackage.jl
```

## Example usage
```
using testpackage
posterior_samples_eirr <- fit_eirrc(data, 
                               obstimes, 
                               param_change_times, 
                               priors_only, 
                               n_samples, 
                               n_chains, 
                               seed)
```
