
# LEFT = 1
# RIGHT = 2
# UP = 3
# DOWN = 4

Base.@kwdef mutable struct GridWorld
    size::Tuple{Int,Int} = (1, 1)
    P::Array{Float64,3} = zeros(1, 1, 4)
    γ::Float64 = 0.99
    Actions = [1, 2, 3, 4]
end

GridWorld(size::Tuple{Int,Int}, γ::Float64) = begin
    env = GridWorld(size=size, γ=γ)
    construct_P(env)
    env
end

function find_neighbors(grid::GridWorld, state::Integer)
    rows, cols = grid.size
    states = rows * cols
    neighbors = []
    for (n, a) ∈ [(state - 1, 1), (state + 1, 2), (state - rows, 3), (state + rows, 4)]
        if n ∈ 1:states
            push!(neighbors, (n, a))
        end
    end
    neighbors
end


function construct_P(grid::GridWorld)
    rows, cols = grid.size
    states = rows * cols
    actions = grid.Actions
    P::Array{Float64,3} = zeros(states, states, length(actions))
    for si ∈ 1:states
        neighbors = find_neighbors(grid, si)
        num_neighbors = length(neighbors)
        for (n, a) in neighbors
            P[si, n, a] = 1 / num_neighbors
        end
    end
    grid.P = P
end




env = GridWorld((2, 2), 0.99)
P = env.P
P[1, :, 1]
find_neighbors(env, 1)