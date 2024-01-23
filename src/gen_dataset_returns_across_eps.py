from typing import List
import numpy as np
from driving_sim import DrivingSim
from gridworld import GridWorld
import pandas as pd
from rmdp import MDP
from concurrent.futures import ProcessPoolExecutor as Pool

def run_one_experiment(env: MDP, eps: float, D: List[List[tuple[int,int]]]) -> dict[str, float]:
    returns_dict: dict[str, float] = {}
    try:
        returns_dict["ROIL_LINF"] = env.solve_cheb_part_2(D, False, True, eps)[3]
    except:
        returns_dict["ROIL_LINF"] = np.nan
    try:
        returns_dict["ROIL_LINF_LIN"] = env.solve_cheb_part_2(D, True, True, eps)[3]
    except:
        returns_dict["ROIL_LINF_LIN"] = np.nan
    returns_dict["Epsilon"] = eps
    returns_dict["Optimal"] = env.opt_return
    returns_dict["Random"] = env.random_return
    return returns_dict

class Experiment:
    """This class is soley for the pickling required by the multiprocessing pool
    it remembers the env for the following call to run_experiments_then_average"""

    def __init__(self, env: MDP, D: List[List[tuple[int,int]]]):
        self.env = env
        self.D = D

    def __call__(self, eps: float):
        return run_one_experiment(self.env, eps, self.D)

def gen_dataset_for_eps(env: MDP, name:str, off_policy: bool):
    D: List[List[tuple[int,int]]] = [[]]
    if off_policy:
        D = env.generate_off_policy_demonstrations(6, 166, env.u_rand)
    else:
        D = env.generate_samples_from_policy(6, 166, env.opt_policy)
    returns_list: dict[str, List[float]] = {}
    closure = Experiment(env, D)
    u_e_hat = env.u_hat_all(D)
    true_eps: float = float(np.linalg.norm(((env.u_E).reshape((env.num_states*env.num_actions), order="F") - u_e_hat.reshape((env.num_states*env.num_actions), order="F"))@env.phi, ord=np.inf))
    print(f"True Epsilon: {true_eps}")
    true_eps *= 1.5 # 50% more than true epsilon because I want to see what happens
    _, syed_rad, syed_ret = env.solve_syed(D, len(D), len(D[0])) # Solve Syed's LPAL once
    with Pool() as pool:
        results = pool.map(closure, np.linspace(100,true_eps,64, dtype=float))
        for result in results:
            result["True_Epsilon"] = true_eps/1.5 # normalize true epsilon
            result["LPAL_Rad"] = syed_rad # Should be the same for all epsilons
            result["LPAL"] = syed_ret
            print(result)
            for key, value in result.items():
                returns_list.setdefault(key, [])
                returns_list[key].append(value)
    dataset = pd.DataFrame.from_dict(returns_list)
    num_rows = int(np.sqrt(env.num_states))
    dataset.to_csv(f"./datasets/epsilon_experiment/{num_rows}x{num_rows}_{name}_{'off' if off_policy else 'on'}_policy.csv", index=False)

def main():
    # create an env
    env_size = 40
    env = GridWorld(env_size,0.99)
    print(f"Running experiment with {env_size}x{env_size} gridworld! Off Policy")
    gen_dataset_for_eps(env, "GridWorld", True)
    print(f"Running experiment with {env_size}x{env_size} gridworld! On Policy")
    gen_dataset_for_eps(env, "GridWorld", False)
    obstacles = list(np.random.choice((env_size*env_size) - (env_size + 1), env_size, replace=False))
    env = DrivingSim(env_size,obstacles)
    print(f"Running experiment with {env_size}x{env_size} driving! Off Policy")
    gen_dataset_for_eps(env, "DrivingSim", True)
    print(f"Running experiment with {env_size}x{env_size} driving! On Policy")
    gen_dataset_for_eps(env, "DrivingSim", False)


if __name__ == "__main__":
    main()
