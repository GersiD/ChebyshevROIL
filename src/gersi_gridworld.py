from typing import List
from gridworld import GridWorld
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool
import matplotlib.pyplot as plt

"""This document is an offshoot of the gridworld document for the uses of Gersi Doko"""


def get_returns_across_methods(env: GridWorld, num_examples: int) -> dict[str, float]:
    """Gets the returns across the hardcoded methods below, then returns a dictionary of the methods name corresponding with its return
    env = An MDP env
    num_examples = the number of demonstrations for methods which require some dataset D ~ P
    """
    # Generate the dataset for all methods to use for this run
    num_episodes = int(np.log(num_examples))
    horizon = num_examples // num_episodes
    D = env.generate_demonstrations_from_occ_freq(env.u_E, num_episodes, horizon)

    # find u for different methods
    returns_list: dict[str, float] = {}

    # add the returns of each u to the list
    returns_list["Chebyshev"] = env.solve_chebyshev_center(D)[2]
    returns_list["LPAL"] = env.solve_syed(D, num_episodes, horizon)[2]

    # optimal return
    returns_list["Optimal"] = env.opt_return
    # random returns
    returns_list["Random"] = env.random_return
    return returns_list


def run_experiments_then_average(
    env: GridWorld, trials: int = 3, episode_len: int = 10
):
    runing_total: dict[str, float] = get_returns_across_methods(env, episode_len)

    for _ in range(1, trials):
        one_experiment_return: dict[str, float] = get_returns_across_methods(
            env, episode_len
        )
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

    def __init__(self, env: GridWorld):
        self.env = env

    def __call__(self, dataset_size: int):
        return run_experiments_then_average(self.env, 3, int(dataset_size))


def plot_experiments_across_dataset_size(env: GridWorld):
    """Given an enviornment this function plots the return of IRL methods asyncronusly"""
    closure = Experiment(env)
    returns_per_DS_size: dict[str, List[float]] = {}
    with Pool(8) as pool:
        results = pool.map(closure, np.linspace(10, 100000, 10, dtype=int))
        for result in results:
            print(result)
            for key, value in result.items():
                returns_per_DS_size.setdefault(key, [])
                returns_per_DS_size[key].append(value)

    # We start plotting
    dataset_size = returns_per_DS_size.pop("dataset_size", None)
    for key, value in returns_per_DS_size.items():
        plt.plot(dataset_size, value, label=key)
    plt.xlabel("Dataset Size")
    plt.ylabel("Average Return")
    plt.title(f"{env.num_rows} x {env.num_rows} Gridworld Experiment gamma={env.gamma}")
    plt.legend()
    plt.savefig(f"plots/{env.num_rows}x{env.num_rows}_gridworld.png")
    # plt.show()


def main():
    # np.random.seed(603)
    exp_size = 20
    env = GridWorld(exp_size, 0.99)
    print(f"Running experiment with {exp_size}x{exp_size} gridworld!")
    plot_experiments_across_dataset_size(env)


if __name__ == "__main__":
    main()
