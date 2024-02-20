from gridworld import GridWorld
from driving_sim import DrivingSim
import numpy as np


def test_cheb_all_states_gridworld():
    env = GridWorld(30, 0.99)
    D = [env.generate_all_expert_demonstrations()]
    (_,_,cheb_return) = env.solve_chebyshev_center(D)
    assert(abs(cheb_return - env.opt_return) <= 1e-8)

def test_cheb2_all_states_gridworld():
    env = GridWorld(30, 0.99)
    D = [env.generate_all_expert_demonstrations()]
    (_,_,_,cheb_return) = env.solve_cheb_part_2(D, True, False)
    assert(abs(cheb_return - env.opt_return) <= 1e-8)

def test_cheb_all_states_driving():
    num_rows = 30
    obstacles = list(np.random.choice((num_rows*num_rows) - (num_rows),  num_rows, replace=False))
    env = DrivingSim(num_rows, obstacles)
    D = [env.generate_all_expert_demonstrations()]
    (_,_,cheb_return) = env.solve_chebyshev_center(D)
    assert(abs(cheb_return - env.opt_return) <= 1e-4) # TODO: Why is the driving sim so much more numerically unstable?

def test_cheb2_all_states_driving():
    num_rows = 30
    obstacles = list(np.random.choice((num_rows*num_rows) - (num_rows),  num_rows, replace=False))
    env = DrivingSim(num_rows, obstacles)
    D = [env.generate_all_expert_demonstrations()]
    (_,_,_,cheb_return) = env.solve_cheb_part_2(D, True, False)
    assert(abs(cheb_return - env.opt_return) <= 1e-4)

def test_cheb2_given_LPAL_RAD():
    num_rows = 20
    obstacles = list(np.random.choice((num_rows*num_rows) - (num_rows),  num_rows, replace=False))
    env = DrivingSim(num_rows, obstacles)
    D = env.generate_samples_from_policy(10, 100, env.opt_policy)
    _, rad, lpal_ret = env.solve_syed(D, 10, 100)
    (eps,_,_,cheb_return) = env.solve_cheb_part_2(D, False, True, rad)
    assert(abs(eps - rad) <= 1e-4)
    assert(abs(cheb_return - lpal_ret) <= 1e-4)


