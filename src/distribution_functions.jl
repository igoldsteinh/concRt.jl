
# credit to Damon Bayer for all of these helper functions
"""
create re-parametrized negative binomial in terms of mean and overdispersion
"""
function NegativeBinomial2(μ, ϕ)
  p = 1 / (1 + μ / ϕ)

  if p <= zero(p)
    p = eps(zero(p))
  end

  if p >= one(p)
    p = prevfloat(one(p))
  end

  r = ϕ

  if r <= zero(r)
    r = eps(zero(r))
  end

  Distributions.NegativeBinomial(r, p)
end

"""
create generalized t distribution with location and scale
"""

function GeneralizedTDist(μ, σ, ν)
  if ν <= zero(ν)
    ν = nextfloat(zero(ν))
  end

  if σ <= zero(σ)
    σ = nextfloat(zero(σ))
  end

  Generalized_T = μ + σ*Distributions.TDist(ν)

  return(Generalized_T)

end

