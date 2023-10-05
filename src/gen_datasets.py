from typing import List
from gridworld import GridWorld
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool
import pandas as pd

"""This document is an offshoot of the gridworld document for the uses of Gersi Doko"""
"""Mainly for running experiments and generating datasets (csv)"""


def get_returns_across_methods(env:GridWorld, D: List[List[tuple[int,int]]], num_episodes: int, horizon: int) -> dict[str, float]:
    """Gets the returns across the hardcoded methods below, then returns a dictionary of the methods name corresponding with its return
    D = a list of demonstrations, where each demonstration is a list of tuples (s,a)
    env = An MDP env
    num_episodes = the number of episodes in D
    horizon = the length of each demonstration
    """
    # find u for different methods
    returns_list: dict[str, float] = {}

    # add the returns of each u to the list
    returns_list["Chebyshev"] = env.solve_chebyshev_center(D)[2]
    returns_list["LPAL"] = env.solve_syed(D, num_episodes, horizon)[2]
    returns_list["GAIL"] = env.solve_GAIL(D, num_episodes, horizon)
    returns_list["BC"] = env.solve_BC(D, num_episodes, horizon)
    u_e_hat, u_e_hat_return = env.solve_naive_BC(D, num_episodes, horizon)
    returns_list["NBC"] = u_e_hat_return
    returns_list["EstLInfDiff"] = np.linalg.norm(env.u_E - u_e_hat, ord=np.inf)
    # optimal return
    returns_list["Optimal"] = env.opt_return
    # returns_list["Worst"] = env.worst_return
    # random returns
    returns_list["Random"] = env.random_return
    return returns_list

def run_one_experiment(env: GridWorld, num_examples: int, off_policy: bool = False) -> dict[str, float]:
    """Runs one experiment for the given environment and number of examples, 
    env = GridWorld environment
    num_examples = the number of demonstrations for methods which require some dataset D 
    off_policy = whether to run the experiment off policy or not
    returns a dictionary of the methods name corresponding with its returns"""
    num_episodes = int(np.log(num_examples))
    horizon = num_examples // num_episodes
    if off_policy:
        D = env.generate_off_policy_demonstrations(num_episodes, horizon, env.u_E, env.u_rand)
        return get_returns_across_methods(env, D, num_episodes, horizon)
    else:
        D = env.generate_demonstrations_from_occ_freq(env.u_E, num_episodes, horizon)
        return get_returns_across_methods(env, D, num_episodes, horizon)

def run_experiments_then_average(
    env: GridWorld, trials: int, episode_len: int, off_policy: bool) -> dict[str, float]:
    runing_total: dict[str, float] = run_one_experiment(env, episode_len, off_policy)

    for _ in range(1, trials):
        one_experiment_return: dict[str, float] = run_one_experiment(env, episode_len, off_policy)
        # Add up the returns
        for key in one_experiment_return.keys():
            runing_total[key] += one_experiment_return[key]

    # Now compute the average for each
    for key in runing_total.keys():
        runing_total[key] /= trials
    runing_total["dataset_size"] = episode_len
    return runing_total

class Experiment:
    """This class is soley for the pickling required by the multiprocessing pool
    it remembers the env for the following call to run_experiments_then_average"""

    def __init__(self, env: GridWorld, off_policy: bool):
        self.env = env
        self.off_policy = off_policy

    def __call__(self, dataset_size: int):
        return run_experiments_then_average(self.env, 1, int(dataset_size), self.off_policy)

def generate_dataset(env: GridWorld, off_policy: bool):
    """Given an enviornment this function generates the csvs of the return of IRL methods asyncronusly"""
    closure = Experiment(env, off_policy)
    returns_per_DS_size: dict[str, List[float]] = {}
    with Pool() as pool:
        results = pool.map(closure, np.linspace(10, 10000, 50, dtype=int))
        for result in results:
            print(result)
            for key, value in result.items():
                returns_per_DS_size.setdefault(key, [])
                returns_per_DS_size[key].append(value)
    dataset = pd.DataFrame.from_dict(returns_per_DS_size)
    dataset.to_csv(f"datasets/{env.num_rows}x{env.num_rows}_gridworld_{'off' if off_policy else 'on'}_policy.csv", index=False)

def generate_datasets_across_env_size(env_sizes: List[int]):
    for env_size in env_sizes:
        print(f"Running experiment with {env_size}x{env_size} gridworld!")
        env = GridWorld(env_size, 0.99)
        generate_dataset(env, True)
        generate_dataset(env, False)

def main():
    np.random.seed(3)
    # if you want to run one experiment
    # exp_size = 5
    # env = GridWorld(exp_size, 0.99)
    # print(f"Running experiment with {exp_size}x{exp_size} gridworld!")
    # generate_dataset(env, False)
    
    # if you want to run multiple experiments
    generate_datasets_across_env_size([5, 10, 20, 30, 40])

if __name__ == "__main__":
    main()
