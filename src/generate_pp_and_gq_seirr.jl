"""
    generate_pp_and_gq_seirr(data, ...)
Transform samples from fit_seirr to useable dataframes 

default priors are for scenario 1, and assume the model is being fit to a daily time scale

# Arguments 
-`samples`: output from fit_seirr
-`data::Float64`: Log RNA concentrations  
-`obstimes::Float64`: times cases are observed
-`param_change_times::Float64`: times when the reproduction number is allowed to change
-`extra_ode_precision::Boolean`: if true, uses custom ode precisions, otherwise uses default values 
-`priors_only::Boolean`: if true function produces draws from the joint prior distribution
-`fit_abs_tol::Float64 = 1e-9`: if `extra_ode_precision` true, absolute tolerance for model fitting 
-`fit_rel_tol::Float64 = 1e-6`: if `extra_ode_precision` true, relative tolerance for model fitting 
-`seed::Int64 = 1`: random seed 
-`popsize::Int64 = 100000`: population size
-`active_pop::Int64 = 90196`: population size - initial size of R compartment
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
-`S_SEIR1_sd::Float64 = 0.05`: standard deviation for normal prior on logit fraction of `active_pop` initially in S
-`S_SEIR1_mean::Float64 = 3.468354`: mean for normal prior on logit fraction of `active_pop` initially in S
-`I_EIR1_sd::Float64 = 0.05`: standard deviation for normal prior on logit fraction of initial E,I and R1 compartments in the I compartment 
-`I_EIR1_mean::Float64 = -1.548302`: mean for normal prior on logit fraction of initial E,I and R1 compartments in the I compartment 
-`R1_ER1_sd::Float64 = 0.05`: standard deviation for normal prior on logit fraction of initial E and R1 compartments in the R1 compartment 
-`R1_ER1_mean::Float64 = 2.221616`: mean for normal prior on logit fraction of initial E and R1 compartments in the R1 compartment 
-`sigma_R0_sd::Float64 = 0.2`: standard deviation for normal prior of log sigma R0
-`sigma_R0_mean::Float64 = log(0.1)`: mean for normal prior of log sigma R0
-`r0_init_sd::Float64 = 0.1`: standard deviation for normal prior of log R0
-`r0_init_mean::Float64 = log(0.88)`: mean for normal prior of log R0
-`lambda_mean::Float64 = 5.685528`: mean for normal prior of logit lambda 
-`lambda_sd::Float64 = 2.178852`: standard deviation for normal prior of logit lambda 
-`df_shape::Float64 = 2.0`: shape parameter for gamma prior of df
-`df_scale::Float64 = 10.0`: scale parameter for gamma prior of df 

"""

function generate_pp_and_gq_seirr(samples, 
  data_cases,
  obstimes,
  param_change_times,
  extra_ode_precision,
  seed::Int64 = 1,
  fit_abs_tol::Float64 = 1e-9,
  fit_rel_tol::Float64 = 1e-6,
  popsize::Int64 = 100000,
  active_pop::Int64 = 92271,
  gamma_sd::Float64 = 0.2,
  gamma_mean::Float64 =log(1/4),
  nu_sd::Float64 = 0.2,
  nu_mean::Float64 = log(1/7),
  eta_sd::Float64 = 0.2,
  eta_mean::Float64 = log(1/18),
  rho_gene_sd::Float64 =  1.0,
  rho_gene_mean::Float64 = 0.0,
  tau_sd::Float64 = 1.0,
  tau_mean::Float64 = 0.0,
  sigma_R0_sd::Float64 = 0.2,
  sigma_R0_mean::Float64 = log(0.1),
  S_SEIR1_sd::Float64 = 0.05,
  S_SEIR1_mean::Float64 = 3.468354,
  I_EIR1_sd::Float64 = 0.05,
  I_EIR1_mean::Float64 = -1.548302,
  R1_ER1_sd::Float64 = 0.05,
  R1_ER1_mean::Float64 = 2.221616,
  r0_init_sd::Float64 = 0.2,
  r0_init_mean::Float64 = log(0.88),
  lambda_mean::Float64 = 5.685528,
  lambda_sd::Float64 = 2.178852,
  df_shape::Float64 = 2.0,
  df_scale::Float64 = 10.0)

  obstimes = convert(Vector{Float64}, obstimes)
  prob = ODEProblem{true}(seir_ode_log!,
  zeros(5),
  (0.0, obstimes[end]),
  ones(3))


  my_model = bayes_seirr(
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
                        eta_sd,
                        eta_mean,
                        rho_gene_sd,
                        rho_gene_mean,
                        tau_sd,
                        tau_mean,
                        sigma_R0_sd,
                        sigma_R0_mean,
                        S_SEIR1_sd,
                        S_SEIR1_mean,
                        I_EIR1_sd,
                        I_EIR1_mean,
                        R1_ER1_sd,
                        R1_ER1_mean,
                        r0_init_sd,
                        r0_init_mean,
                        lambda_mean,
                        lambda_sd,
                        df_shape,
                        df_scale)


  missing_data = repeat([missing], length(data))

  my_model = bayes_seirr(
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
                        eta_sd,
                        eta_mean,
                        rho_gene_sd,
                        rho_gene_mean,
                        tau_sd,
                        tau_mean,
                        sigma_R0_sd,
                        sigma_R0_mean,
                        S_SEIR1_sd,
                        S_SEIR1_mean,
                        I_EIR1_sd,
                        I_EIR1_mean,
                        R1_ER1_sd,
                        R1_ER1_mean,
                        r0_init_sd,
                        r0_init_mean,
                        lambda_mean,
                        lambda_sd,
                        df_shape,
                        df_scale)


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
