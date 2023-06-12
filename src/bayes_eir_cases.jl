"""
    bayes_eir_closed!(outs_tmp, ...)
Turing model for the EIR model. 

default priors are for scenario 1, and assume the model is being fit to a weekly time scale

# Arguments 
-`outs_tmp`: preallocated matrix used to store ODE solutions, created using the dualcache function from https://github.com/SciML/PreallocationTools.jl
-`data_cases::Int64`: Counts of cases 
-`obstimes::Float64`: times cases are observed
-`param_change_times::Float64`: times when the reproduction number is allowed to change
-`priors_only::Boolean`: if TRUE function produces draws from the joint prior distribution
-`n_samples::Int64 = 250`: number of posterior samples AFTER Burn-in, total samples will be twice `n_samples`
-`n_chains::Int64 = 4`: number of chains 
-`seed::Int64 = 1`: random seed 
-`gamma_sd::Float64 = 0.2`: standard deviation for normal prior of log gamma 
-`gamma_mean::Float64 =log(1/4)`: mean for normal prior of log gamma 
-`nu_sd::Float64 = 0.2`: standard deviation for normal prior of log nu
-`nu_mean::Float64 = log(1/7)`: mean for normal prior of log nu
-`rho_case_sd::Float64= 1.0`: standard devation for normal prior of log rho 
-`rho_case_mean::Float64 = 0.0`: mean for normal prior of log rho 
-`phi_sd::Float64 = 1.0`: standard deviation for normal prior of log phi
-`phi_mean::Float64 = 0.0`: mean for normal prior of log phi 
-`I_init_sd::Float64 = 0.05`: standard deviation for normal prior of I_init
-`I_init_mean::Float64 = 489.0`: mean for normal prior of I_init 
-`E_init_sd::Float64 = 0.05`: standard deviation for normal prior of E_init 
-`E_init_mean::Float64 = 225.0`: mean for normal prior of E_init 
-`sigma_rt_sd::Float64 = 0.2`: standard deviation for normal prior on log sigma rt 
-`sigma_rt_mean::Float64 = log(0.1)`: mean for normal prior on log sigma rt 
-`rt_init_sd::Float64 = 0.1`: standard deviation for normal prior on log rt_init 
-`rt_init_mean::Float64 = log(0.88)`: mean for normal prior on log rt_init 

"""

@model function bayes_eir_closed!(outs_tmp, 
                                 data_cases, 
                                 obstimes, 
                                 param_change_times,
                                 gamma_sd::Float64 = 0.2,
                                 gamma_mean::Float64 =log(1/4),
                                 nu_sd::Float64 = 0.2,
                                 nu_mean::Float64 = log(1/7),
                                 rho_case_sd::Float64= 1.0,
                                 rho_case_mean::Float64 = 0.0,
                                 phi_sd::Float64 = 0.2,
                                 phi_mean::Float64 = log(50),
                                 I_init_sd::Float64 = 0.05,
                                 I_init_mean::Float64 = 489.0,
                                 E_init_sd::Float64 = 0.05,
                                 E_init_mean::Float64 = 225.0,
                                 sigma_rt_sd::Float64 = 0.2,
                                 sigma_rt_mean::Float64 = log(0.1),
                                 rt_init_sd::Float64 = 0.1,
                                 rt_init_mean::Float64 = log(0.88))
  # Calculate number of observed datapoints timepoints
  l_copies = length(obstimes)
  l_param_change_times = length(param_change_times)

  # Priors
  rt_params_non_centered ~ MvNormal(zeros(l_param_change_times + 2), Diagonal(ones(l_param_change_times + 2))) # +2, 1 for var, 1 for init
  gamma_non_centered ~ Normal() # rate to I
  nu_non_centered ~ Normal() # rate to Re
  I_init_non_centered ~ Normal()
  E_init_non_centered ~ Normal()
  rho_case_non_centered ~ Normal() # case detection rate
  phi_non_centered ~ Normal() # NB overdispersion

  # Transformations
  gamma = exp(gamma_non_centered * gamma_sd + gamma_mean)
  nu = exp(nu_non_centered * nu_sd + nu_mean)
  rho_case = logistic(rho_case_non_centered * rho_case_sd + rho_case_mean)
  
  phi_cases = exp(phi_non_centered * phi_sd + phi_mean)

  sigma_rt_non_centered = rt_params_non_centered[1]

  sigma_rt = exp(sigma_rt_non_centered * sigma_rt_sd + sigma_rt_mean)

  rt_init_non_centered = rt_params_non_centered[2]

  rt_init = exp(rt_init_non_centered * rt_init_sd + rt_init_mean)

  alpha_init = rt_init * nu

  log_rt_steps_non_centered = rt_params_non_centered[3:end]

  I_init = I_init_non_centered * I_init_sd + I_init_mean
  E_init = E_init_non_centered * E_init_sd + E_init_mean
  C_init = I_init 
  u0 = [E_init, I_init, 1.0, C_init] 

  # Time-varying parameters
  alpha_t_values_no_init = exp.(log(rt_init) .+ cumsum(vec(log_rt_steps_non_centered) * sigma_rt)) * nu
  alpha_t_values_with_init = vcat(alpha_init, alpha_t_values_no_init)
  sol_reg_scale_array = eir_closed_solution!(outs_tmp, 1:obstimes[end], param_change_times, 0.0, alpha_t_values_with_init, u0, gamma, nu)
  if any(!isfinite, sol_reg_scale_array)
    Turing.@addlogprob! -Inf
    return
  end


  new_cases = sol_reg_scale_array[5, 2:end] - sol_reg_scale_array[5, 1:(end-1)]
  cases_mean = new_cases .* rho_case
  for i in 1:round(Int64, l_copies)
    data_cases[i] ~ NegativeBinomial2(max(cases_mean[i], 0.0), phi_cases)
  end
# Generated quantities
  rt_t_values_with_init = alpha_t_values_with_init/ nu

  return (
    gamma = gamma,
    nu = nu,
    rho_cases = rho_case,
    rt_init = rt_init,
    sigma_rt = sigma_rt,
    phi_cases = phi_cases,
    alpha_t_values = alpha_t_values_with_init,
    rt_t_values = rt_t_values_with_init,
    I_init,
    E_init,
    E = sol_reg_scale_array[2, :],
    I = sol_reg_scale_array[3, :],
    R = sol_reg_scale_array[4, :],
    new_cases = new_cases,
    cases_mean = cases_mean
  )
end
