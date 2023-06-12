"""
    eir_closed_solution!(outs_tmp, times, change_times, t0, alphas, init_conds, gamma, nu)
Solve the EIR ODE system at `times` beginning from `t0` 

Alpha changes at time `change_times` to the corresponding value of `alphas`
# Arguments 
-`outs_tmp`: preallocated matrix used to store ODE solutions, created using the dualcache function from https://github.com/SciML/PreallocationTools.jl
-`times::float64`: times to save the ODE solution 
-`t0::float64`: intitial time
-`alphas::float64`: values of alpha at corresponding values of `times`
-`init_conds::float64`: initial conditions of the compartments at `t0`
-`gamma:float64`: inverse of gamma is (loosely) the average time spent in the E compartment 
-`nu::float64`: inverse of nu is (loosely) the average time spent in the I compartment
"""

function eir_closed_solution!(outs_tmp, times, change_times, t0, alphas, init_conds, gamma, nu) 
    num_alphas = length(alphas)
    stop_times = vcat(change_times, times[end])
    start_times = vcat(times[1], change_times .+ 1.0)
    outs_matrix = get_tmp(outs_tmp, [alphas[1], gamma, nu])
    current_init_conds = init_conds
    current_init_time = t0
    first_column = vcat(t0, init_conds)
    # first column is time
    # next four columns are the solutions
    for i in 1:num_alphas
      alpha = alphas[i]
      current_stop = stop_times[i]
      current_start = start_times[i]
      for j in current_start:current_stop
        t = j - current_init_time
  
        @inbounds begin 
        outs_matrix[1, round(Int64, j)] = j

        #2nd row 

        outs_matrix[2, round(Int64, j)] = current_init_conds[1] *  (-((exp(((-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*
          (2*alpha*gamma + power(gamma,2) - gamma*nu - gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
        (-4*alpha*gamma - power(gamma,2) + 2*gamma*nu - power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
          nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
     (exp(((-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*
        (2*alpha*gamma + power(gamma,2) - gamma*nu + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
        nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) +
        current_init_conds[2] *  (-((exp(((-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*
          (-(alpha*gamma) - alpha*nu + alpha*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
        (-4*alpha*gamma - power(gamma,2) + 2*gamma*nu - power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
          nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
     (exp(((-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*
        (-(alpha*gamma) - alpha*nu - alpha*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
        nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))

        # 3rd row 
        outs_matrix[3, round(Int64,j)] = current_init_conds[1] * (-((exp(((-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*
          (-power(gamma,2) - gamma*nu + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
        (-4*alpha*gamma - power(gamma,2) + 2*gamma*nu - power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
          nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
     (exp(((-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*
        (-power(gamma,2) - gamma*nu - gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
        nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) +
        current_init_conds[2] * (-((exp(((-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*
          (2*alpha*gamma - gamma*nu + power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
        (-4*alpha*gamma - power(gamma,2) + 2*gamma*nu - power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
          nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
     (exp(((-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*
        (2*alpha*gamma - gamma*nu + power(nu,2) + nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
        nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))

        # 4th row 
        outs_matrix[4, round(Int64, j)] = current_init_conds[1] *  (-(nu/(alpha - nu)) - (2*exp(((-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*gamma*nu)/
      (-4*alpha*gamma - power(gamma,2) + 2*gamma*nu - power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
        nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) + 
     (2*exp(((-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*gamma*nu)/
      (4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
        nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
        current_init_conds[2] * (-(nu/(alpha - nu)) + (exp(((-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*nu*
        (-gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (-4*alpha*gamma - power(gamma,2) + 2*gamma*nu - power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
        nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) - 
     (exp(((-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*t)/2)*nu*
        (-gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2) + gamma*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)) + 
        nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
        current_init_conds[3] * (((-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
       (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/(4*gamma*(alpha - nu)))


       # 5th row
       outs_matrix[5, round(Int64, j)] =  current_init_conds[1] * (-(nu/(alpha - nu)) + (exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
       (-2*alpha*gamma + gamma*nu - power(nu,2) + nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
     (2*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) + 
    (exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
       (2*alpha*gamma - gamma*nu + power(nu,2) + nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
     (2*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
     current_init_conds[2] * (-(alpha/(alpha - nu)) + (alpha*exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
       (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
     (2*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) + 
    (alpha*exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
       (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
     (2*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
     current_init_conds[4]
  
  
        end 
      end 
      current_init_conds = outs_matrix[2:5,round(Int64, current_stop)]
      current_init_time = outs_matrix[1, round(Int64, current_stop )]
    end
  
    outs_matrix = hcat(first_column, outs_matrix)
    return(outs_matrix)
  end 

