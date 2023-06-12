"""
    optimize_many_MAP(model, n_reps = 100, top_n = 1, verbose = true)

Try n_reps different initializations to get MAP estimate.

Function by Damon Bayer

"""
function optimize_many_MAP(model, n_reps = 100, top_n = 1, verbose = true)
  lp_res = repeat([-Inf], n_reps)
  for i in eachindex(lp_res)
      if verbose
          println(i)
      end
      Random.seed!(i)
      try
          lp_res[i] = optimize(model, MAP(), LBFGS(linesearch = LineSearches.BackTracking())).lp
      catch
      end
  end
  eligible_indices = findall(.!isnan.(lp_res) .& isfinite.(lp_res))
  best_n_seeds =  eligible_indices[sortperm(lp_res[eligible_indices], rev = true)][1:top_n]

  map(best_n_seeds) do seed
    Random.seed!(seed)
    optimize(model, MAP(), LBFGS(linesearch = LineSearches.BackTracking())).values.array
  end
end
