using LinearAlgebra
using MultivariateStats
using Plots
using QHull
using StatsBase
using JuMP
using LazySets
import HiGHS

# U is a vector of matricies
# each u ∈ U is a matrix of SxA
# u is indexed by [s, a]
function construct_U(S, A, P, γ, p₀)
    U = Matrix{Real}[]
    # Putterman State-action frequency,  pgs. 398-404
    for π ∈ collect(Base.product(A, A, A)) #enumerate all Policies
        # Solve linear system for d
        K = cat(P[1, :, π[1]], P[2, :, π[2]], P[3, :, π[3]], dims=2)'
        d = (I - γ * K') \ p₀

        # u(s,a) = d[s] * pi[s,a]
        u = zeros((length(S), length(A)))
        for a ∈ A # Assuming |A| uniform across states
            u[:, a] = d .* (π .== a)
        end
        push!(U, u)
    end
    U
end

# Compute if s was observed in D with some action a
function observed(s, D)
    for (state, _) ∈ D
        if (state == s)
            return true
        end
    end
    return false
end

# Construnct the constraint c which constrains 
# some u ∈ U to be in Υ
function construct_constraint_vector(S, A, D)
    # Define the constraint vector
    c = zeros(length(S), length(A))
    for state in S
        for action in A
            if ((state, action) ∉ D && observed(state, D))
                c[state, action] = 1
            end
        end

    end
    vec(c) # turn c back into a vector iterating by state first
end

# Υ is a matrix with SA rows
# each col υ ∈ Υ is a occupancy frequency
function construct_upsilon(UM, S, A, D)
    Υ = [] # Set of deterministic occupancy frequencies consistent with D
    c = construct_constraint_vector(S, A, D) # Constraint vector
    for u in eachcol(UM)
        if c' * u == 0
            push!(Υ, vec(u)) # Add the consistent occ freq to Upsilon
        end
    end
    reduce(hcat, Υ)
end

# Construct a dataset D of tuples (s, aₑ) 
# Given a policy πₑ
function construct_dataset(dLen, πₑ, S, P, p₀)
    D = []
    s = StatsBase.sample(S, Weights(p₀), 1)[1]
    for t in 1:dLen
        a = πₑ(s)
        push!(D, (s, a))
        s = StatsBase.sample(S, Weights(vec(P[s, :, a])), 1)[1]
    end
    D
end

# Wrapper around the chull library
# This function mainly controls what happens
# when there are only 1 or 2 elements of x,y 
function calculate_convex_hull(x, y)
    simplices = []
    if length(x) == 2
        simplices = [(1, 2), (2, 1)] # Convex hull of 2 points is a line
    elseif length(x) == 1
        simplices = [(1, 1)] # Convex hull of 1 point is the point itself
    else
        ch = chull(hcat(x, y))
        simplices = ch.simplices
    end
    simplices
end

# Plotting function whick plots a convex hull 
# set is a two row matrix of points [x ; y]
# draw_interior controls whether or not to plot the innner points of the hull
function draw_convex_hull(set, plot, color, label, draw_interior=true)
    x = set[1, :]
    y = set[2, :]
    simplices = calculate_convex_hull(x, y)

    draw_interior && scatter!(plot, x, y, marker=:circle, label=label, color=color) # fancy one liner
    for (i, j) ∈ simplices
        plot!(plot, [x[i], x[j]], [y[i], y[j]], color=color, label=nothing)
    end
end

# Function to construct the matrix of constraints for the set U
# Here we stack each (I-γPₐ') on top of eachother
# A = action space
# P = transition matrix
# γ = discount factor
function construct_design_matrix(A, P, γ)
    # Define the design matrix
    tmp = []
    for a in A
        push!(tmp, (I - γ * P[:, :, a])) # Notice we are pushing matrices onto tmp
    end
    # Now to reduce tmp from a vector of matrices to a matrix we concatinate
    # the elements of tmp horizontally and then transpose them
    reduce(hcat, tmp')'
end

# This function solves for the chebyshev center of Upsilon
# Φ = feature marix SA X k
# S = state space
# A = action space
# p₀ = initial state distribution
# γ = discount factor
# P = transition matrix
# D = observed dataset of (s,a) pairs
function solve_cheb(Φ, S, A, p₀, γ, P, D)
    _, num_features = size(Φ)
    model = Model(HiGHS.Optimizer)
    set_silent(model) # keeps the solver quiet
    W = construct_design_matrix(A, P, γ)
    c = construct_constraint_vector(S, A, D)
    @show c

    s = length(S)
    a = length(A)
    sa = s * a
    @variable(model, σ >= 0)
    @variable(model, u[1:sa] .≥ 0)
    @variable(model, α[1:num_features])
    @variable(model, α̂[1:num_features])
    @variable(model, β[1:s, 1:num_features])
    @variable(model, β̂[1:s, 1:num_features])

    # @constraint(model, [i=1:num_features], -β[:,i]'p₀ .≤ σ + u'*Φ[:,i])
    # @constraint(model, [i=1:num_features], -β̂[:,i]'p₀ .≤ σ - u'*Φ[:,i])
    # @constraint(model, [i=1:num_features], α[i]*c + W*β[:,i] .<= -Φ[:,i])
    # @constraint(model, [i=1:num_features], α̂[i]*c + W*β̂[:,i] .<= Φ[:,i])

    @constraint(model, [i = 1:num_features], σ + Φ[:, i]' * u .>= p₀' * β[:, i])
    @constraint(model, [i = 1:num_features], σ - Φ[:, i]' * u .>= p₀' * β̂[:, i])
    @constraint(model, [i = 1:num_features], W * β[:, i] + α[i] * c .>= Φ[:, i])
    @constraint(model, [i = 1:num_features], W * β̂[:, i] + α̂[i] * c .>= -1 * Φ[:, i])

    @constraint(model, c0, W' * u .== p₀)
    # @constraint(model, c1, u' * c .== 0) # constrain u to be in Upsilon

    @objective(model, Min, σ)
    optimize!(model)
    @show value.(u)' * c
    @show value.(α)
    has_values(model) || error("Optimizer error: Did not compute a solution.")
    return (value(σ), value.(u))
end