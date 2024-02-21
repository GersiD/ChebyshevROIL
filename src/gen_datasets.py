from typing import List
from gridworld import GridWorld
from driving_sim import DrivingSim
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool
import pandas as pd
from rmdp import MDP
import pickle

"""This document is an offshoot of the gridworld document for the uses of Gersi Doko"""
"""Mainly for running experiments and generating datasets (csv)"""


def get_returns_across_methods(env:MDP, D: List[List[tuple[int,int]]], num_episodes: int, horizon: int) -> dict[str, float]:
    """Gets the returns across the hardcoded methods below, then returns a dictionary of the methods name corresponding with its return
    D = a list of demonstrations, where each demonstration is a list of tuples (s,a)
    env = An MDP env
    num_episodes = the number of episodes in D
    horizon = the length of each demonstration
    """
    # find u for different methods
    returns_list: dict[str, float] = {}

    # add the returns of each u to the list
    _, lpal_rad, lpal_ret = env.solve_syed(D, num_episodes, horizon)
    _, lin_lpal_rad, lin_lpal_ret = env.solve_syed(D, num_episodes, horizon, add_lin_constr=True)
    returns_list["LPAL"] = lpal_ret
    returns_list["LPAL_LIN"] = lin_lpal_ret
    returns_list["ROIL_LIN"] = env.solve_cheb_part_2(D, True, False)[3]
    eps, _, _, ret = env.solve_cheb_part_2(D, False, True, 1.5*lin_lpal_rad)
    returns_list["ROIL_LINF"] = ret
    returns_list["ROIL_LINF_LIN"] = env.solve_cheb_part_2(D, True, True, 1.5*lin_lpal_rad)[3]
    returns_list["GAIL"] = env.solve_GAIL(D, num_episodes, horizon)
    returns_list["BC"] = env.solve_BC(D, num_episodes, horizon)
    u_e_hat, u_e_hat_return = env.solve_naive_BC(D, num_episodes, horizon)
    returns_list["NBC"] = u_e_hat_return
    returns_list["EstLInfDiff"] = float(np.linalg.norm(env.u_E - u_e_hat, ord=np.inf))
    returns_list["Epsilon"] = eps
    # optimal return
    returns_list["Optimal"] = env.opt_return
    # returns_list["Worst"] = env.worst_return
    # random returns
    returns_list["Random"] = env.random_return
    D_size = 0
    for d in D:
        D_size += len(d)
    returns_list["dataset_size"] = D_size
    return returns_list

def run_one_experiment(env: MDP, num_examples: int, off_policy: bool = False) -> dict[str, float]:
    """Runs one experiment for the given environment and number of examples, 
    env = GridWorld environment
    num_examples = the number of demonstrations for methods which require some dataset D 
    off_policy = whether to run the experiment off policy or not
    returns a dictionary of the methods name corresponding with its returns"""
    num_episodes = int(np.log(num_examples))
    horizon = num_examples // num_episodes
    if off_policy:
        D = env.generate_off_policy_demonstrations(num_episodes, horizon, env.u_rand)
        return get_returns_across_methods(env, D, num_episodes, horizon)
    else:
        D = env.generate_samples_from_policy(num_episodes, horizon, env.opt_policy)
        return get_returns_across_methods(env, D, num_episodes, horizon)

def run_experiments_then_average(
    env: MDP, trials: int, episode_len: int, off_policy: bool) -> List[dict[str, float]]:
    runing_total: List[dict] = []

    for _ in range(0, trials):
        one_experiment_return: dict[str, float] = run_one_experiment(env, episode_len, off_policy)
        runing_total.append(one_experiment_return)

    return runing_total

class Experiment:
    """This class is soley for the pickling required by the multiprocessing pool
    it remembers the env for the following call to run_experiments_then_average"""

    def __init__(self, env: MDP, monolith_dataset: List[List[tuple[int,int]]], off_policy: bool):
        self.env = env
        self.off_policy = off_policy
        self.monolith_dataset = monolith_dataset

    def __call__(self, it: int):
        return run_experiments_then_average(self.env, 10, it, self.off_policy)

def generate_dataset(env: MDP, off_policy: bool, name: str):
    """Given an enviornment this function generates the csvs of the return of IRL methods asyncronusly"""
    monolith_dataset: List[List[tuple[int, int]]]= [[]]
    closure = Experiment(env, monolith_dataset, off_policy)
    returns_per_DS_size: dict[str, List[float]] = {}
    results = []
    with Pool() as pool:
        results = pool.map(closure, np.linspace(100,10000,12, dtype=int))
        for resList in results:
            print(resList)
            for result in resList:
                for key, value in result.items():
                    returns_per_DS_size.setdefault(key, [])
                    returns_per_DS_size[key].append(value)

    dataset = pd.DataFrame.from_dict(returns_per_DS_size)
    num_rows = int(np.sqrt(env.num_states))
    dataset.to_csv(f"datasets/{num_rows}x{num_rows}_{name}_{'off' if off_policy else 'on'}_policy.csv", index=False)

def generate_datasets_across_env_size(env_sizes: List[int]):
    for env_size in env_sizes:
        print(f"Running experiment with {env_size}x{env_size} gridworld!")
        obstacles = list(np.random.choice((env_size*env_size) - (env_size + 1), env_size, replace=False))
        env = DrivingSim(env_size, obstacles)
        # with open(f"envs/{env_size}x{env_size}_driving_env.pkl", "wb") as f:
        #     pickle.dump(env, f)
        generate_dataset(env, True, "driving")
        generate_dataset(env, False, "driving")
        env = GridWorld(env_size, 0.99)
        # with open(f"envs/{env_size}x{env_size}_gridworld_env.pkl", "wb") as f:
        #     pickle.dump(env, f)
        generate_dataset(env, True, "gridworld")
        generate_dataset(env, False, "gridworld")

def main():
    # np.random.seed(3)
    # if you want to run one experiment
    # exp_size = 5
    # env = GridWorld(exp_size, 0.99)
    # print(f"Running experiment with {exp_size}x{exp_size} gridworld!")
    # generate_dataset(env, False)
    
    # if you want to run multiple experiments
    generate_datasets_across_env_size([5,10,20,30,40])

if __name__ == "__main__":
    main()
