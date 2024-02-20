import numpy as np
from driving_sim import DrivingSim

def test_reward():
    num_rows = 10
    obstacles = list(np.random.choice((num_rows*num_rows) - (num_rows),  num_rows, replace=False))
    env = DrivingSim(num_rows, obstacles)
    assert(np.all(env.reward >= 0))
