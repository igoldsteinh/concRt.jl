function generate_pp_and_gq_seir(samples, 
  data_cases,
  obstimes,
  param_change_times,
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


  missing_data = repeat([missing], length(data))

  my_model_forecast_missing = bayes_seir_cases(
    missing_data, 
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
