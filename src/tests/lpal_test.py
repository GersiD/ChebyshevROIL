from gridworld import GridWorld
from driving_sim import DrivingSim
import numpy as np

def test_gridworld_true_u_e_hat():
    env = GridWorld(10, 0.99)
    D = env.generate_samples_from_policy(1, 100, env.opt_policy)
    (_, _, syed_return) = env.solve_syed(D, 1, 100, env.u_E_flat)
    assert(abs(syed_return - env.opt_return) <= 1e-8)

def test_driving_true_u_e_hat():
    num_rows = 10
    obstacles = list(np.random.choice((num_rows*num_rows) - (num_rows),  num_rows, replace=False))
    env = DrivingSim(num_rows, obstacles)
    D = env.generate_samples_from_policy(1, 100, env.opt_policy)
    (_, _, syed_return) = env.solve_syed(D, 1, 100, env.u_E_flat)
    assert(abs(syed_return - env.opt_return) <= 1e-8)
