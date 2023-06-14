"""
    fit_eirrc_closed(data, ...)
Fit the EIRR model to observed wastewater concentrations to produce a posterior estimate of the effective reproduction number.

default priors are for scenario 1, and assume the model is being fit to a daily time scale

# Arguments 
-`data::Float64`: Log RNA concentrations
-`obstimes::Float64`: times RNA concentrations are observed
-`param_change_times::Float64`: times when the reproduction number is allowed to change
-`priors_only::Boolean`: if true function produces draws from the joint prior distribution
-`n_samples::Int64 = 250`: number of posterior samples AFTER Burn-in, total samples will be twice `n_samples`
-`n_chains::Int64 = 4`: number of chains 
-`seed::Int64 = 1`: random seed 
-`gamma_sd::Float64 = 0.2`: standard deviation for normal prior of log gamma 
-`gamma_mean::Float64 =log(1/4)`: mean for normal prior of log gamma 
-`nu_sd::Float64 = 0.2`: standard deviation for normal prior of log nu
-`nu_mean::Float64 = log(1/7)`: mean for normal prior of log nu
-`eta_sd::Float64 = 0.2`: standard deviation for normal prior of log eta 
-`eta_mean::Float64 = log(1/18)`: mean for normal prior of log eta 
-`rho_gene_sd::Float64= 1.0`: standard devation for normal prior of log rho 
-`rho_gene_mean::Float64 = 0.0`: mean for normal prior of log rho 
-`tau_sd::Float64 = 1.0`: standard deviation for normal prior of log tau
-`tau_mean::Float64 = 0.0`: mean for normal prior of log tau 
-`I_init_sd::Float64 = 0.05`: standard deviation for normal prior of I_init
-`I_init_mean::Float64 = 489.0`: mean for normal prior of I_init 
-`R1_init_sd::Float64 = 0.05`: standard deviation for normal prior of R1_init 
-`R1_init_mean::Float64 = 2075.0`: mean for normal prior of R1_init 
-`E_init_sd::Float64 = 0.05`: standard deviation for normal prior of E_init 
-`E_init_mean::Float64 = 225.0`: mean for normal prior of E_init 
-`lambda_mean::Float64 = 5.685528`: mean for normal prior of logit lambda 
-`lambda_sd::Float64 = 2.178852`: standard deviation for normal prior of logit lambda 
-`df_shape::Float64 = 2.0`: shape parameter for gamma prior of df
-`df_scale::Float64 = 10.0`: scale parameter for gamma prior of df 
-`sigma_rt_sd::Float64 = 0.2`: standard deviation for normal prior on log sigma rt 
-`sigma_rt_mean::Float64 = log(0.1)`: mean for normal prior on log sigma rt 
-`rt_init_sd::Float64 = 0.1`: standard deviation for normal prior on log rt_init 
-`rt_init_mean::Float64 = log(0.88)`: mean for normal prior on log rt_init 

"""
function fit_eirrc_closed(data,
                          obstimes,
                          param_change_times,
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
