"""
    generate_pp_and_gq_seir(samples, ...)
Transform samples from fit_seir into useable dataframes. 

default priors are for scenario 1, and assume the model is being fit to a weekly time scale

# Arguments 
-`samples`: output from fit_seir 
-`data_cases::Int64`: Counts of cases 
-`obstimes::Float64`: times cases are observed
-`param_change_times::Float64`: times when the reproduction number is allowed to change
-`extra_ode_precision::Boolean`: if true, uses custom ode precisions, otherwise uses default values 
-`fit_abs_tol::Float64 = 1e-9`: if `extra_ode_precision` true, absolute tolerance for model fitting 
-`fit_rel_tol::Float64 = 1e-6`: if `extra_ode_precision` true, relative tolerance for model fitting 
-`opt_abs_tol::Float64 = 1e-11`: if `extra_ode_precision` true, absolute tolerance for choosing mcmc initial values 
-`opt_rel_tol::Float64 = 1e-8`: if `extra_ode_precision` true, relative tolerance for choosing mcmc initial values
-`popsize::Int64 = 100000`: population size
-`active_pop::Int64 = 90196`: population size - initial size of R compartment
-`seed::Int64 = 1`: random seed 
-`gamma_sd::Float64 = 0.2`: standard deviation for normal prior of log gamma 
-`gamma_mean::Float64 =log(1/4)`: mean for normal prior of log gamma 
-`nu_sd::Float64 = 0.2`: standard deviation for normal prior of log nu
-`nu_mean::Float64 = log(1/7)`: mean for normal prior of log nu
-`rho_case_sd::Float64= 1.0`: standard devation for normal prior of log rho 
-`rho_case_mean::Float64 = 0.0`: mean for normal prior of log rho 
-`phi_sd::Float64 = 1.0`: standard deviation for normal prior of log phi
-`phi_mean::Float64 = 0.0`: mean for normal prior of log phi 
-`S_SEI_sd::Float64 = 0.05`: standard deviation for normal prior of logit fraction of active pop initially in S
-`S_SEI_mean::Float64 = 4.83091`: mean for normal prior of logit fraction of active pop initially in S
-`I_EI_sd::Float64 = 0.05`: standard deviation for normal prior of logit fraction of the E and I compartments initially in I
-`I_EI_mean::Float64 = 0.7762621`: mean for normal prior of logit fraction of the E and I compartments initially in I
-`sigma_R0_sd::Float64 = 0.2`: standard deviation for normal prior of log sigma R0
-`sigma_R0_mean::Float64 = log(0.1)`: mean for normal prior of log sigma R0
-`r0_init_sd::Float64 = 0.1`: standard deviation for normal prior of log R0
-`r0_init_mean::Float64 = log(0.88)`: mean for normal prior of log R0

"""

function generate_pp_and_gq_seir(samples, 
  data_cases,
  obstimes,
  param_change_times,
  extra_ode_precision,
  seed::Int64 = 1,
  fit_abs_tol::Float64 = 1e-9,
  fit_rel_tol::Float64 = 1e-6,
  popsize::Int64 = 100000,
  active_pop::Int64 = 90196,
  gamma_sd::Float64 = 0.2,
  gamma_mean::Float64 =log(7/4),
  nu_sd::Float64 = 0.2,
  nu_mean::Float64 = log(7/7),
  rho_case_sd::Float64 =  0.4,
  rho_case_mean::Float64 = -1.386294,                  
  phi_sd::Float64 = 0.2,
  phi_mean::Float64 = log(50),
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



  my_model = bayes_seir(
    data_cases, 
    obstimes, 
    param_change_times, 
    extra_ode_precision, 
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
    phi_sd,
    phi_mean,
    sigma_R0_sd,
    sigma_R0_mean,
    S_SEI_sd,
    S_SEI_mean,
    I_EI_sd,
    I_EI_mean,
    r0_init_sd,
    r0_init_mean)


  missing_data = repeat([missing], length(data))

  my_model_forecast_missing = bayes_seir(
    missing_data, 
    obstimes, 
    param_change_times, 
    extra_ode_precision, 
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
    phi_sd,
    phi_mean,
    sigma_R0_sd,
    sigma_R0_mean,
    S_SEI_sd,
    S_SEI_mean,
    I_EI_sd,
    I_EI_mean,
    r0_init_sd,
    r0_init_mean)

  # remove samples which are NAs
  indices_to_keep = .!isnothing.(generated_quantities(my_model, samples));

  samples_randn = ChainsCustomIndex(samples, indices_to_keep);


  Random.seed!(seed)
  predictive_randn = predict(my_model_forecast_missing, samples_randn)

  Random.seed!(seed)
  gq_randn = Chains(generated_quantities(my_model, samples_randn))

  samples_df = DataFrame(samples)

  results = [DataFrame(predictive_randn), DataFrame(gq_randn), samples_df]
  return(results)
end
