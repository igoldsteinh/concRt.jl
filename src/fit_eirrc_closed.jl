using Turing
using Random
using ForwardDiff
using Logging
using PreallocationTools
Logging.disable_logging(Logging.Warn)
## Define closed form solution
# include(projectdir("src/closed_soln_eirr_withincid.jl"))
# ## Load Model
# include(projectdir("src/bayes_eirrc_closed.jl"))
# ## Load extra distributions
# include(projectdir("src/distribution_functions.jl"))
# ## Load extra distributions
# include(projectdir("src/helper_functions.jl"))

# function for fitting the eirr model with latent cumulative incidence
# data is assumed to be a vector of observed log RNA concentrations
# obstimes is assumed to be the corresponding vectors of observation times
# obstimes should start with 1 being the first observed time
# default priors are for scenario 1, and assume the model is being fit to a daily time scale
# this can be adjusted by rescaling the rate parameters
# gamma_sd::Float64 = 0.2
# gamma_mean::Float64 =log(1/4)
# nu_sd::Float64 = 0.2
# nu_mean::Float64 = log(1/7)
# eta_sd::Float64 = 0.2
# eta_mean::Float64 = log(1/18)
# rho_gene_sd::Float64= 1.0
# rho_gene_mean::Float64 = 0.0
# tau_sd::Float64 = 1.0
# tau_mean::Float64 = 0.0
# I_init_sd::Float64 = 0.05
# I_init_mean::Float64 = 489.0
# R1_init_sd::Float64 = 0.05
# R1_init_mean::Float64 = 2075.0
# E_init_sd::Float64 = 0.05
# E_init_mean::Float64 = 225.0
# lambda_mean::Float64 = 5.685528
# lambda_sd::Float64 = 2.178852
# df_shape::Float64 = 2.0
# df_scale::Float64 = 10.0
# sigma_rt_sd::Float64 = 0.2
# sigma_rt_mean::Float64 = log(0.1)
# rt_init_sd::Float64 = 0.1
# rt_init_mean::Float64 = log(0.88)
# data = data_log_copies
# obstimes = long_dat[:, :new_time]
# priors_only = false
# n_samples::Int64 = 10
# n_chains::Int64 = 1
# seed::Int64 = 1
function fit_eirrc_closed(data,
                          obstimes,
                          priors_only,
                          n_samples::Int64 = 250,
                          n_chains::Int64 = 4,
                          seed::Int64 = 1,
                          gamma_sd::Float64 = 0.2,
                          gamma_mean::Float64 =log(1/4),
                          nu_sd::Float64 = 0.2,
                          nu_mean::Float64 = log(1/7),
                          eta_sd::Float64 = 0.2,
                          eta_mean::Float64 = log(1/18),
                          rho_gene_sd::Float64= 1.0,
                          rho_gene_mean::Float64 = 0.0,
                          tau_sd::Float64 = 1.0,
                          tau_mean::Float64 = 0.0,
                          I_init_sd::Float64 = 0.05,
                          I_init_mean::Float64 = 489.0,
                          R1_init_sd::Float64 = 0.05,
                          R1_init_mean::Float64 = 2075.0,
                          E_init_sd::Float64 = 0.05,
                          E_init_mean::Float64 = 225.0,
                          lambda_mean::Float64 = 5.685528,
                          lambda_sd::Float64 = 2.178852,
                          df_shape::Float64 = 2.0,
                          df_scale::Float64 = 10.0,
                          sigma_rt_sd::Float64 = 0.2,
                          sigma_rt_mean::Float64 = log(0.1),
                          rt_init_sd::Float64 = 0.1,
                          rt_init_mean::Float64 = log(0.88))
  obstimes = convert(Vector{Float64}, obstimes)
  # trying to avoid the stupid situation where we're telling to change at the end of the solver which doesn't make sense
  if maximum(obstimes) % 7 == 0
    param_change_max = maximum(obstimes) - 7
  else
    param_change_max = maximum(obstimes)
  end
  param_change_times = collect(7:7.0:param_change_max)
  outs_tmp = dualcache(zeros(6,length(1:obstimes[end])), 10)


  my_model = bayes_eirrc_closed!(
      outs_tmp,
      data,
      obstimes,
      param_change_times,
      gamma_sd,
      gamma_mean,
      nu_sd,
      nu_mean,
      eta_sd,
      eta_mean,
      rho_gene_sd,
      rho_gene_mean,
      tau_sd,
      tau_mean,
      I_init_sd,
      I_init_mean,
      R1_init_sd,
      R1_init_mean,
      E_init_sd,
      E_init_mean,
      lambda_mean,
      lambda_sd,
      df_shape,
      df_scale,
      sigma_rt_sd,
      sigma_rt_mean,
      rt_init_sd,
      rt_init_mean)

  # Sample Posterior

  if priors_only
    Random.seed!(seed)
    samples = sample(my_model, Prior(), MCMCThreads(), 400, n_chains)
  else
    Random.seed!(seed)

    MAP_init = optimize_many_MAP(my_model, 10, 1, false)[1]

    Random.seed!(seed)
    MAP_noise = vcat(randn(length(MAP_init) - 1, n_chains), transpose(zeros(n_chains)))
    MAP_noise = [MAP_noise[:,i] for i in 1:size(MAP_noise,2)]

    init = repeat([MAP_init], n_chains) .+ 0.05 * MAP_noise

    Random.seed!(seed)
    samples = sample(my_model, NUTS(), MCMCThreads(), n_samples, n_chains, discard_initial = n_samples, init_params = init)
  end

  return(samples)
end
