from typing import List
from gridworld import GridWorld
from driving_sim import DrivingSim
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool
import pandas as pd
from rmdp import MDP
import pickle
import os
import itertools

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
    # _, lpal_rad, lpal_ret = env.solve_syed(D, num_episodes, horizon)
    # _, lin_lpal_rad, lin_lpal_ret = env.solve_syed(D, num_episodes, horizon, add_lin_constr=True)
    # returns_list["LPAL"] = lpal_ret
    # returns_list["LPAL_LIN"] = lin_lpal_ret
    # _, _, _, cheb_lin_ret = env.solve_cheb_part_2(D, add_lin_constr=True, add_linf_constr=False)
    # returns_list["ROIL_LIN"] = cheb_lin_ret
    # eps, _, _, cheb_linf_ret = env.solve_cheb_part_2(D, add_lin_constr=False, add_linf_constr=True, passed_eps=1.5*lin_lpal_rad)
    # returns_list["ROIL_LINF"] = cheb_linf_ret
    # returns_list["ROIL_LINF_LIN"] = env.solve_cheb_part_2(D, add_lin_constr=True, add_linf_constr=True, passed_eps=1.5*lin_lpal_rad)[3]
    # returns_list["ROIL_LINF_PRUNE"] = env.solve_cheb_part_2(D, add_lin_constr=False, add_linf_constr=True, passed_eps=1.5*lin_lpal_rad, prune=True)[3]
    # returns_list["ROIL_LIN_PRUNE"] = env.solve_cheb_part_2(D, add_lin_constr=True, add_linf_constr=False, prune=True)[3]
    # returns_list["GAIL"] = env.solve_GAIL(D, num_episodes, horizon)[0]
    # returns_list["BC"] = env.solve_BC(D, num_episodes, horizon)
    # u_e_hat, u_e_hat_return = env.solve_naive_BC(D, num_episodes, horizon)
    # returns_list["NBC"] = u_e_hat_return
    # returns_list["EstLInfDiff"] = float(np.linalg.norm(env.u_E - u_e_hat, ord=np.inf))
    # returns_list["Epsilon"] = eps
    # # optimal return
    # returns_list["Optimal"] = env.opt_return
    # returns_list["Worst"] = env.worst_return
    # random returns
    # returns_list["Random"] = env.random_return
    # D_flat = set(itertools.chain.from_iterable(D))
    # returns_list["S_Cover"] = ((len(D_flat) / env.num_states) * 100)
    # returns_list["LIN_REG"] = env.worst_case_regret(D, env.solve_cheb_part_2(D, add_lin_constr=True, add_linf_constr=False)[1].reshape((env.num_states*env.num_actions), order="F"))
    # returns_list["ROIL_P_REG"] = env.worst_case_regret(D, env.solve_cheb_part_2(D, add_lin_constr=True, add_linf_constr=False, prune=True)[1].reshape(env.num_states*env.num_actions, order="F"))
    # returns_list["NBC_REG"] = env.worst_case_regret(D, env.solve_naive_BC(D, num_episodes, horizon)[0].reshape((env.num_states*env.num_actions), order="F"))
    # returns_list["GAIL_REG"] = env.worst_case_regret(D, env.solve_GAIL(D, num_episodes, horizon)[1].reshape((env.num_states*env.num_actions), order="F"))
    # returns_list["LPAL_REG"] = env.worst_case_regret(D, env.solve_syed(D, num_episodes, horizon)[0].reshape((env.num_states*env.num_actions), order="F"))
    # returns_list["OPT_REG"] = env.worst_case_regret(D, env.u_E_flat)
    # returns_list["RAND_REG"] = env.worst_case_regret(D, env.u_rand.reshape((env.num_states*env.num_actions), order="F"))
    # returns_list["WORST_REG"] = env.worst_case_regret(D, env.worst_u.reshape((env.num_states*env.num_actions), order="F"))

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

class OneExperiment:
    """This class is soley for the pickling required by the multiprocessing pool"""
    def __init__(self, env: MDP, num_examples: int, off_policy: bool):
        self.env = env
        self.num_examples = num_examples
        self.off_policy = off_policy

    def __call__(self, *args, **kwargs):
        return run_one_experiment(self.env, self.num_examples, self.off_policy)

def run_experiments_then_average(
    env: MDP, trials: int, episode_len: int, off_policy: bool) -> List[dict[str, float]]:
    runing_total: List[dict] = []

    for _ in range(0, trials):
        one_experiment_return: dict[str, float] = run_one_experiment(env, episode_len, off_policy)
        runing_total.append(one_experiment_return)
    # closure = OneExperiment(env, episode_len, off_policy) # Currently do not have enough memory
    # with Pool(5) as pool:
    #     runing_total = list(pool.map(closure, range(trials)))

    return runing_total

class Experiment:
    """This class is soley for the pickling required by the multiprocessing pool
    it remembers the env for the following call to run_experiments_then_average"""

    def __init__(self, env: MDP, off_policy: bool):
        self.env = env
        self.off_policy = off_policy

    def __call__(self, it: int):
        return run_experiments_then_average(self.env, 5, it, self.off_policy)

def generate_dataset(env: MDP, off_policy: bool, name: str):
    """Given an enviornment this function generates the csvs of the return of IRL methods asyncronusly"""
    closure = Experiment(env, off_policy)
    returns_per_DS_size: dict[str, List[float]] = {}
    results = []
    with Pool(10) as pool:
        results = pool.map(closure, np.linspace(100,10000,12, dtype=int))
        for resList in results:
            # print(resList)
            for result in resList:
                for key, value in result.items():
                    returns_per_DS_size.setdefault(key, [])
                    returns_per_DS_size[key].append(value)
    num_rows = int(np.sqrt(env.num_states))
    file_name = f"datasets/{num_rows}x{num_rows}_{name}_{'off' if off_policy else 'on'}_policy.csv"
    try: # if the file exists, append to it
        df = pd.read_csv(file_name)
        print(f"\033[31m{file_name} already exists, opening...\033[0m")
        keys = list(returns_per_DS_size.keys())
        for key in keys:
            # add or replace column as long as it's not the dataset size
            if key != "dataset_size":
                print(f"\033[33mReplacing {key} in {file_name}\033[0m") if key in df else print(f"\033[32mAdding {key} to {file_name}\033[0m")
                df[key] = returns_per_DS_size[key]
                del returns_per_DS_size[key]
        df.to_csv(file_name, index=False)
    except FileNotFoundError:
        print(f"\033[31mCreating {file_name}\033[0m")
        dataset = pd.DataFrame.from_dict(returns_per_DS_size)
        dataset.to_csv(file_name, index=False)

def generate_datasets_across_env_size(env_sizes: List[int]):
    for env_size in env_sizes:
        print(f"Running experiment with {env_size}x{env_size} gridworld!")
        with open(f"envs/{env_size}x{env_size}_driving_env.pkl", "rb") as f:
            env = pickle.load(f)
            generate_dataset(env, True, "driving")
            generate_dataset(env, False, "driving")
        with open(f"envs/{env_size}x{env_size}_gridworld_env.pkl", "rb") as f:
            env = pickle.load(f)
            generate_dataset(env, True, "gridworld")
            generate_dataset(env, False, "gridworld")

def main():
    # Check if envs have been generated before
    sizes = [5,10,20,30,40]
    for size in sizes:
        try:
            with open(f"envs/{size}x{size}_gridworld_env.pkl", "rb") as f:
                pass
        except FileNotFoundError:
            env = GridWorld(size, 0.8)
            print(f"\033[31m{size}x{size}_gridworld_env.pkl not found, generating...\033[0m")
            with open(f"envs/{size}x{size}_gridworld_env.pkl", "wb") as f:
                pickle.dump(env, f)
        try:
            with open(f"envs/{size}x{size}_driving_env.pkl", "rb") as f:
                pass
        except FileNotFoundError:
            print(f"\033[31m{size}x{size}_driving_env.pkl not found, generating...\033[0m")
            obstacles = list(np.random.choice((size*size) - (size + 1), size, replace=False))
            env = DrivingSim(size, obstacles)
            with open(f"envs/{size}x{size}_driving_env.pkl", "wb") as f:
                pickle.dump(env, f)
    # if you want to run multiple experiments
    generate_datasets_across_env_size(sizes)

if __name__ == "__main__":
    main()
