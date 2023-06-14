"""
    fit_eir_closed(data_cases, ...)
Fit the EIR model to observed case counts to produce a posterior estimate of the effective reproduction number.

default priors are for scenario 1, and assume the model is being fit to a weekly time scale

# Arguments 
-`data_cases::Int64`: Counts of cases 
-`obstimes::Float64`: times cases are observed
-`param_change_times::Float64`: times when the reproduction number is allowed to change
-`priors_only::Boolean`: if true function produces draws from the joint prior distribution
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

function fit_eir_closed(data_cases,
                          obstimes,
                          param_change_times,
                          priors_only,
                          n_samples::Int64 = 250,
                          n_chains::Int64 = 4,
                          seed::Int64 = 1,
                          gamma_sd::Float64 = 0.2,
                          gamma_mean::Float64 =log(7/4),
                          nu_sd::Float64 = 0.2,
                          nu_mean::Float64 = log(7/7),
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
    obstimes = convert(Vector{Float64}, obstimes)


    outs_tmp = dualcache(zeros(5,length(1:obstimes[end])), 10)

        

    my_model = bayes_eir_closed!(outs_tmp, 
                                data_cases, 
                                obstimes, 
                                param_change_times,
                                gamma_sd,
                                gamma_mean,
                                nu_sd,
                                nu_mean,
                                rho_case_sd,
                                rho_case_mean,
                                phi_sd,
                                phi_mean,
                                I_init_sd,
                                I_init_mean,
                                E_init_sd,
                                E_init_mean,
                                sigma_rt_sd,
                                sigma_rt_mean,
                                rt_init_sd,
                                rt_init_mean)

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