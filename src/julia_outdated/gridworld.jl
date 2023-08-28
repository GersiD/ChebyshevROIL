function compute_features(colors::Int, num_states::Int, num_actions::Int)
  # Compute the features for the gridworld
  # returns an SA X K matrix where K is the number of colors
  phi_s = zeros(Float64, num_states, colors)
  # Pick a random color for each state
  indices = rand(1:colors, num_states)
  for i in 1:num_states
    phi_s[i, indices[i]] = 1
  end
  phi = reduce(vcat, [phi_s for i in 1:num_actions])
  @show rank(phi) == colors # is the matrix singular?
  return phi
end

function compute_reward(colors::Int, num_states::Int, num_actions::Int, phi::Matrix{Float64})
  signs = rand([-1, 1], colors)
  w = signs .* (range(1, colors))
  phi_s = phi[1:num_states, :]
  rewards_s = (phi_s * w)
  rewards_s /= norm(rewards_s, 1)
  # display(rewards_s)
  rewards = reduce(vcat, [rewards_s for i in 1:num_actions])
  w = phi \ rewards
  @show norm(w, 1) # should be less than 1
  return rewards
end

function compute_transition(num_states::Int, num_actions::Int)
  # Compute the transition matrix for the gridworld
  # returns an S X S X A matrix indexed by [current state, next state, action]
  num_rows = Int(sqrt(num_states))
  P = zeros(Float64, num_states, num_states, num_actions)
  p1, p2 = 0.2, 0.2
  action_prob = p1 * ones(num_actions, num_actions) + p2 * I

  for a in 1:num_actions
    for s in 1:num_states
      if s % num_rows != 0 # can go right
        P[s, s+1, a] += action_prob[a, 1]
      else
        P[s, s, a] += action_prob[a, 1]
      end
      if s > num_rows # can go up
        P[s, s-num_rows, a] += action_prob[a, 2]
      else
        P[s, s, a] += action_prob[a, 2]
      end
      if (s - 1) % num_rows != 0 # can go left
        P[s, s-1, a] += action_prob[a, 3]
      else
        P[s, s, a] += action_prob[a, 3]
      end
      if s <= (num_states - num_rows) # can go down
        P[s, s+num_rows, a] += action_prob[a, 4]
      else
        P[s, s, a] += action_prob[a, 4]
      end
    end
  end

  for s in 1:num_states
    for a in 1:num_actions
      @assert (sum(P[s, :, a]) - 1) < 1e-6
    end
  end
  return P
end


