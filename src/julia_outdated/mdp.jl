include("gridworld.jl")
using StatsBase
using LinearAlgebra
using JuMP
import HiGHS

struct MDP <: Any
  states::Vector{Int}
  actions::Vector{Int}
  num_states::Int
  num_actions::Int
  num_features::Int
  phi::Matrix{Float64}
  P::Array{Float64,3} # indexed by [cur_state, next_state, action]
  p₀::Vector{Float64} # S vector
  reward::Vector{Float64} #S x A vector
  γ::Float64
  W::Matrix{Float64} # design matrix

  # Constructor
  function MDP(states::Vector{Int}, actions::Vector{Int}, features::Array{Float64,2}, gamma::Real, reward::Vector{Float64}, P::Array{Float64,3}, p₀::Vector{Float64})
    num_states = length(states)
    num_actions = length(actions)
    num_features = length(features)
    W = construct_design_matrix(actions, P, gamma)
    new(states, actions, num_states, num_actions, num_features, features, P, p₀, reward, gamma, W)
  end
  # Gridworld Constructor
  function MDP(num_rows::Int, γ::Real)
    num_states = num_rows^2
    num_actions = 4
    states = collect(1:num_states)
    actions = collect(1:num_actions)
    k = 4
    phi = compute_features(k, num_states, num_actions)
    rewards = compute_reward(k, num_states, num_actions, phi)
    P = compute_transition(num_states, num_actions)
    p₀ = ones(num_states) / num_states
    new(states, actions, num_states, num_actions, k, phi, P, p₀, rewards, γ, construct_design_matrix(actions, P, γ))
  end

end

function next_state(mdp::MDP, state, action)
  # Sample next state from transition probabilities
  next_state = sample(mdp.states, Weights(mdp.P[state, :, action], 1))
  return next_state
end

function occupancy_frequency_to_policy(mdp::MDP, occ_freq::Array{Float64})
  # Convert state X action occupancy frequencies to a policy
  # occ_freq is indexed by [state, action]
  policy = zeros(mdp.num_states, mdp.num_actions)
  for state in mdp.states
    policy[state, :] = occ_freq[state, :] ./ max(sum(occ_freq[state, :]), 1e-6)
  end
  return policy
end

function construct_design_matrix(actions::Vector{Int}, P::Array{Float64,3}, γ::Float64)
  # Constructs stacked (I - γP) matrices 
  # Also called the design matrix
  tmp = []
  for a in actions
    push!(tmp, (I - γ * P[:, :, a])) # Notice we are pushing matrices onto tmp
  end
  # Now to reduce tmp from a vector of matrices to a matrix we concatinate
  # the elements of tmp horizontally and then transpose them
  # TODO this is really dumb and slow
  reduce(hcat, tmp')'
end

function generate_samples_from_policy(mdp::MDP, policy::Array{Float64,2}, horizon::Int64)
  # Generate a sample trajectory from the given policy
  # policy is indexed by [state, action]
  samples = Array{Tuple{Int,Int},1}(undef, horizon)
  state = sample(mdp.states, Weights(mdp.p₀, 1))
  for t in 1:horizon
    action = sample(mdp.actions, Weights(policy[state, :], 1))
    samples[t] = (state, action)
    state = next_state(mdp, state, action)
  end
  return samples
end

function generate_demonstrations_from_occ_freq(mdp::MDP, occ_freq::Array{Float64,2}, num_episodes::Int64, horizon::Int64)
  # Generate demonstrations from the given state-action occurence frequencies
  # occ_freq is indexed by [state, action]
  # return a 2D array of tuples, indexed by [episode, timestep]
  @show Threads.nthreads()
  demonstrations = Array{Tuple{Int,Int},2}(undef, num_episodes, horizon)
  policy = occupancy_frequency_to_policy(mdp, occ_freq)
  Threads.@threads for i in 1:num_episodes
    demonstrations[i, :] = generate_samples_from_policy(mdp, policy, horizon)
  end
  return demonstrations
end

function solve_putterman_dual(mdp::MDP)
  method = "Dual_LP"
  s = mdp.num_states
  a = mdp.num_actions
  sa = s * a
  p₀ = mdp.p₀
  γ = mdp.γ
  reward = mdp.reward
  W = mdp.W

  model = Model(HiGHS.Optimizer)
  set_silent(model)

  @variable(model, u[1:sa] .>= 0)
  @constraint(model, W' * u .== p₀)
  @objective(model, Max, u' * reward)

  optimize!(model)
  has_values(model) || error("{} failed to converge", method)
  u = value.(u)
  (sum(u) - (1 / (1 - γ)) < 1e-6) || error("{} did not find an occupancy frequncy", method)
  return (reshape(u, (s, a)), objective_value(model))
end

function observed(s, D)
  for (s, _) in D
    if s == s
      return true
    end
  end
  return false
end

function construct_constraint_vector(mdp::MDP, D::Vector{Tuple{Int,Int}})
  c = zeros(mdp.num_states, mdp.num_actions)
  for s in mdp.states
    observed_s = observed(s, D)
    for a in mdp.actions
      if observed_s && (s, a) ∉ D
        c[s, a] = 1
      end
    end
  end
  vec(c)
end

function solve_cheb(mdp::MDP, num_samples::Int, occ_freq::Matrix{Float64})
  method = "Cheb"
  s = mdp.num_states
  a = mdp.num_actions
  sa = s * a
  p₀ = mdp.p₀
  Φ = mdp.phi
  num_features = mdp.num_features
  γ = mdp.γ
  reward = mdp.reward
  W = mdp.W
  D = generate_demonstrations_from_occ_freq(mdp, occ_freq, 1, num_samples)
  c = construct_constraint_vector(mdp, D[1, :])

  model = Model(HiGHS.Optimizer)
  set_silent(model)

  @variable(model, σ >= 0)
  @variable(model, u[1:sa] .≥ 0)
  @variable(model, α[1:num_features] .>= 0)
  @variable(model, α̂[1:num_features] .>= 0)
  @variable(model, β[1:s, 1:num_features])
  @variable(model, β̂[1:s, 1:num_features])

  @constraint(model, [i = 1:num_features], σ + Φ[:, i]' * u .>= p₀' * β[:, i])
  @constraint(model, [i = 1:num_features], σ - Φ[:, i]' * u .>= p₀' * β̂[:, i])
  @constraint(model, [i = 1:num_features], W * β[:, i] + α[i] * c .>= Φ[:, i])
  @constraint(model, [i = 1:num_features], W * β̂[:, i] + α̂[i] * c .>= -1 * Φ[:, i])

  @constraint(model, W' * u .== p₀) # Constrain u to be a valid occupancy frequency

  @objective(model, Min, σ)

  optimize!(model)
  has_values(model) || error("{} failed to converge", method)
  u = value.(u)
  (sum(u) - (1 / (1 - γ)) < 1e-6) || error("{} did not find an occupancy frequncy", method)
  return (reshape(u, (s, a)), value(σ), u' * reward)
end

# function solve_syed(mdp::MDP, episodes::Int, horizon::Int)
#   method = "Syed"
#   s = mdp.num_states
#   a = mdp.num_actions
#   p₀ = mdp.p₀
#   phi = mdp.features
#   W = mdp.W
#   V_hat = zeros(mdp.num_features)
#   D = generate_demonstrations_from_occ_freq(mdp, , episodes, horizon)
#
#   
#
# end
#

env = MDP(10, 0.99)
u_e, opt_ret = solve_putterman_dual(env)
@show opt_ret
_, cheb_rad, cheb_ret = solve_cheb(env, 10, u_e)
@show cheb_rad
@show cheb_ret