# closed form solution for eirr model 
# note this produces Nans when alpha = nu 
# I'm hoping this can never happen irl but we'll have to see when we test it out
function power(a,b)
  a^b
end 

# current version which works well
function new_eirrc_closed_solution!(outs_tmp, times, change_times, t0, alphas, init_conds, gamma, nu, eta) 
  num_alphas = length(alphas)
  stop_times = vcat(change_times, times[end])
  start_times = vcat(times[1], change_times .+ 1)
  outs_matrix = get_tmp(outs_tmp, [alphas[1], gamma, nu, eta])
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

      outs_matrix[2, round(Int64, j)] = current_init_conds[1] * (-0.25*(exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
      (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
      (-2*alpha*gamma + gamma*nu - power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) - 
    (exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
     (-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (2*alpha*gamma - gamma*nu + power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (4*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
    current_init_conds[2] * ((alpha*exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
     (-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (4*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) + 
    (alpha*exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
     (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (4*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))

        outs_matrix[3, round(Int64, j)] = current_init_conds[1] * (-0.25*(exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
      (-2*alpha*gamma + gamma*nu - power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
      (2*alpha*gamma - gamma*nu + power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (alpha*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) - 
    (exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
     (2*alpha*gamma - gamma*nu + power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (2*alpha*gamma - gamma*nu + power(nu,2) + nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (4*alpha*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
    current_init_conds[2] * ((exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
     (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (2*alpha*gamma - gamma*nu + power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (4*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) + 
    (exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
     (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (2*alpha*gamma - gamma*nu + power(nu,2) + nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (4*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))
    

        outs_matrix[4, round(Int64, j)] = current_init_conds[1] * (-0.25*(exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*nu*
      (gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
      (gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
      (-2*alpha*gamma + gamma*nu - power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (alpha*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))*
      (-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
    (exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*nu*
     (gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (2*alpha*gamma - gamma*nu + power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (4*alpha*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))*
     (-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
    ((2*nu)/(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) - 
     (2*power(nu,2))/(alpha*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
     (2*nu)/(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) + 
     (2*power(nu,2))/(alpha*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))))/
    (exp(eta*t)*(-(sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma) + 
       (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma)))) + 
    current_init_conds[2] * ((exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*nu*
     (gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (4*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))*
     (-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
    (exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*nu*
     (gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (4*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))*
     (-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
    ((nu*(-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (4*alpha*power(gamma,2)*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
     (nu*(gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (4*alpha*power(gamma,2)*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))))/
    (exp(eta*t)*(-(sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma) + 
       (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma)))) +  
    current_init_conds[3] * (-((sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma - 
      (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma))/
    (exp(eta*t)*(-(sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma) + 
        (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma)))))
    

      outs_matrix[5, round(Int64, j)] = current_init_conds[1] * (-0.5*(exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*eta*nu*
      (-gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
      (-2*alpha*gamma + gamma*nu - power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (alpha*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))*
      (-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
    (exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*eta*nu*
     (-gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (2*alpha*gamma - gamma*nu + power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (2*alpha*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))*
     (-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
    ((2*nu)/(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) - 
     (2*power(nu,2))/(alpha*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
     (2*nu)/(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) + 
     (2*power(nu,2))/(alpha*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))))/
    (exp(eta*t)*(-(sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma) + 
       (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma))) + 
    ((-2*nu)/(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) - 
     (eta*nu)/(alpha*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
     (2*power(nu,2))/(alpha*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
     (eta*power(nu,2))/(alpha*gamma*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
     (eta*nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/
      (alpha*gamma*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
     (2*nu)/(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) + 
     (eta*nu)/(alpha*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
     (2*power(nu,2))/(alpha*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
     (eta*power(nu,2))/(alpha*gamma*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
     (eta*nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/
      (alpha*gamma*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))))/
    (-(sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma) + 
     (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma))) + 
     current_init_conds[2] * ((exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*eta*nu*
     (-gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (2*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))*
     (-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
    (exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*eta*nu*
     (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
     (-gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
    (2*gamma*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))*
     (-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
    ((nu*(-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (4*alpha*power(gamma,2)*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) - 
     (nu*(gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
      (4*alpha*power(gamma,2)*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))))/
    (exp(eta*t)*(-(sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma) + 
       (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma))) + 
    (((-gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (-((eta*nu*(-gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
             (alpha*gamma*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))) - 
          (nu*(gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
             (gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
           (2*alpha*gamma*(-2*eta + gamma + nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))))/(2*gamma) - 
     ((-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
        (-((eta*nu*(-gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
             (alpha*gamma*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))) - 
          (nu*(gamma - nu - sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))*
             (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
           (2*alpha*gamma*(-2*eta + gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))))/(2*gamma))/
    (-(sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma) + 
     (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma))) + 
    current_init_conds[3] * (1 + (sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma - 
     (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma))/
    (exp(eta*t)*(-(sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))/gamma) + 
       (nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))/(alpha*gamma)))) +
    current_init_conds[4]    

          outs_matrix[6, round(Int64, j)] = current_init_conds[1] * (-(nu/(alpha - nu)) - (exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
          (-2*alpha*gamma + gamma*nu - power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
         (2*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) - 
         (exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
          (2*alpha*gamma - gamma*nu + power(nu,2) - nu*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
         (2*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
         current_init_conds[2] * (-(alpha/(alpha - nu)) + (alpha*exp(((-gamma - sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
          (-gamma - nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
         (2*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))) + 
         (alpha*exp(((-gamma + sqrt(4*alpha*gamma + power(gamma - nu,2)) - nu)*t)/2)*
          (gamma + nu + sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2))))/
         (2*(alpha - nu)*sqrt(4*alpha*gamma + power(gamma,2) - 2*gamma*nu + power(nu,2)))) + 
         current_init_conds[5]         

      end 
    end 
    current_init_conds = outs_matrix[2:6,round(Int64, current_stop)]
    current_init_time = outs_matrix[1, round(Int64, current_stop )]
  end

  outs_matrix = hcat(first_column, outs_matrix)
  return(outs_matrix)
end 

