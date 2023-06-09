
data_cases = [75, 108, 115, 145, 268, 510, 901, 1178, 2112, 2151, 2212, 1780, 1338, 882, 590, 338, 199, 90, 77]
    obstimes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    param_change_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    priors_only = false
    n_samples::Int64 = 10
    n_chains::Int64 = 1
    fit_abs_tol::Float64 = 1e-9
    fit_rel_tol::Float64 = 1e-6
    opt_abs_tol::Float64 = 1e-11
    opt_rel_tol::Float64 = 1e-8
    seed::Int64 = 1
    popsize::Int64 = 100000
    active_pop::Int64 = 90196
    gamma_sd::Float64 = 0.2
    gamma_mean::Float64 =log(7/4)
    nu_sd::Float64 = 0.2
    nu_mean::Float64 = log(7/7)
    rho_case_sd::Float64 =  0.4
    rho_case_mean::Float64 = -1.386294
    sigma_R0_sd::Float64 = 0.2
    sigma_R0_mean::Float64 = log(0.1)
    S_SEI_sd::Float64 = 0.05
    S_SEI_mean::Float64 = 4.83091
    I_EI_sd::Float64 = 0.05
    I_EI_mean::Float64 = 0.7762621
    r0_init_sd::Float64 = 0.1
    r0_init_mean::Float64 = log(0.88)

include("seir_ode_log.jl")
include("optimize_many_MAP.jl")
include("bayes_seir.jl")
function fit_seir(data_cases,
                  obstimes,
                  param_change_times,
                  priors_only,
                  n_samples::Int64 = 250,
                  n_chains::Int64 = 4,                  
                  fit_abs_tol::Float64 = 1e-9,
                  fit_rel_tol::Float64 = 1e-6,
                  opt_abs_tol::Float64 = 1e-11,
                  opt_rel_tol::Float64 = 1e-8,
                  seed::Int64 = 1,
                  popsize::Int64 = 100000,
                  active_pop::Int64 = 90196,
                  gamma_sd::Float64 = 0.2,
                  gamma_mean::Float64 =log(7/4),
                  nu_sd::Float64 = 0.2,
                  nu_mean::Float64 = log(7/7),
                  rho_case_sd::Float64 =  0.4,
                  rho_case_mean::Float64 = -1.386294,
                  sigma_R0_sd::Float64 = 0.2,
                  sigma_R0_mean::Float64 = log(0.1),
                  S_SEI_sd::Float64 = 0.05,
                  S_SEI_mean::Float64 = 4.83091,
                  I_EI_sd::Float64 = 0.05,
                  I_EI_mean::Float64 = 0.7762621,
                  r0_init_sd::Float64 = 0.1,
                  r0_init_mean::Float64 = log(0.88))

  obstimes = convert(Vector{Float64}, obstimes)
  
  prob = ODEProblem{true}(seir_ode_log!,
  zeros(5),
  (0.0, obstimes[end]),
  ones(3))


  my_model_optimize = bayes_seir_cases(
                                        data_cases, 
                                        obstimes, 
                                        param_change_times, 
                                        true, 
                                        prob,
                                        opt_abs_tol,
                                        opt_rel_tol,
                                        popsize,
                                        active_pop,
                                        gamma_sd,
                                        gamma_mean,
                                        nu_sd,
                                        nu_mean,
                                        rho_case_sd,
                                        rho_case_mean,
                                        sigma_R0_sd,
                                        sigma_R0_mean,
                                        S_SEI_sd,
                                        S_SEI_mean,
                                        I_EI_sd,
                                        I_EI_mean,
                                        r0_init_sd,
                                        r0_init_mean)

  my_model = bayes_seir_cases(
                              data_cases, 
                              obstimes, 
                              param_change_times, 
                              true, 
                              prob,
                              fit_abs_tol,
                              fit_rel_tol,
                              popsize,
                              active_pop,
                              gamma_sd,
                              gamma_mean,
                              nu_sd,
                              nu_mean,
                              rho_case_sd,
                              rho_case_mean,
                              sigma_R0_sd,
                              sigma_R0_mean,
                              S_SEI_sd,
                              S_SEI_mean,
                              I_EI_sd,
                              I_EI_mean,
                              r0_init_sd,
                              r0_init_mean)



  # Sample Posterior

  if priors_only
    Random.seed!(seed)
    samples = sample(my_model, Prior(), MCMCThreads(), 400, n_chains)
  else
    Random.seed!(seed)
                          
    MAP_init = optimize_many_MAP(my_model_optimize, 10, 1, false)[1]
                          
    Random.seed!(seed)
    MAP_noise = vcat(randn(length(MAP_init) - 1, n_chains), transpose(zeros(n_chains)))
    MAP_noise = [MAP_noise[:,i] for i in 1:size(MAP_noise,2)]
                          
    init = repeat([MAP_init], n_chains) .+ 0.05 * MAP_noise
                          
    Random.seed!(seed)
    samples = sample(my_model, NUTS(), MCMCThreads(), n_samples, n_chains, discard_initial = n_samples, init_params = init)
  end
                          
  return(samples)
end 
