from gridworld import GridWorld
from typing import List, Tuple
import numpy as np


class LavaWorld(GridWorld):
    """LavaWorld is a gridworld with lava cells that have negative reward
    and a goal cell that has positive reward. Here we only consider a 5x5 gridworld."""
    def __init__(self, gamma):
        self.num_states = 5*5
        self.num_actions = 4
        self.REWARD_STATE = 20
        super().__init__(5, gamma, self.compute_features(), self.compute_reward(), self.compute_p_0())

    def compute_features(self) -> np.ndarray:
        """Every cell is color 0 except for the goal cell which is color 1. Lava cells are -1."""
        phi_s = np.ones((self.num_states, 1)) * 0
        for state in range(self.num_states):
            if state >= 15 and state <= 18:
                phi_s[state] = -1 # lava
            elif state == self.REWARD_STATE:
                phi_s[state] = 1 # goal
        phi = np.vstack([phi_s for _ in range(self.num_actions)]) 
        return phi

    def compute_reward(self) -> np.ndarray:
        """Normal cells have reward -1, goal cell has reward 100, lava cells have reward -1000."""
        reward_s = np.ones(self.num_states) * -1
        for state in range(self.num_states):
            if state >= 15 and state <=18:
                reward_s[state] = -1000
            elif state == self.REWARD_STATE:
                reward_s[state] = 100
        rewards = np.hstack([reward_s for _ in range(self.num_actions)])
        return rewards

    def compute_p_0(self) -> np.ndarray:
        """You only start in state 10 (zero indexed), draw it out smarty pants"""
        p_0 = np.zeros(self.num_states)
        p_0[10] = 1
        return p_0

def main():
    env = LavaWorld(0.99)
    # D: List[List[Tuple[int,int]]]  = [[(10,3)]]
    D: List[List[Tuple[int,int]]]= [[(10,0), (11,0), (12,0), (13,0), (14, 3), (19,3), (24, 2), (23,2), (22,2), (21,2), (20,3)]] 
    
    print(f"\033[32mOptimal Return = { env.opt_return }\033[0m")
    # print(f"\033[32mOptimal occ_freq = { env.u_E }\033[0m")
    # print(f"\033[32mOptimal policy = { env.occupancy_freq_to_policy(env.u_E) }\033[0m")
    (u_cheb, radius, cheb_return) = env.solve_chebyshev_center(D)
    # cheb_pol = env.occupancy_freq_to_policy(u_cheb)
    print(f"\033[34mCheb Return    = { cheb_return }\033[0m")
    print(f"\033[34mCheb Radius    = { radius }\033[0m")
    # print(f"\033[34mCheb occ_freq = { u_cheb }\033[0m")
    # print(f"\033[34mCheb policy = { env.occupancy_freq_to_policy(u_cheb) }\033[0m")
    (u_syed, radius, syed_return) = env.solve_syed(D, len(D), len(D[0]))
    print(f"Syed Return    = { syed_return }")
    print(f"Syed Radius    = { radius }")
    # print(f"Syed occ_freq = { u_syed }")
    # print(f"Syed policy = { env.occupancy_freq_to_policy(u_syed) }")
    print(f"Random return  = { env.random_return }")

if __name__ == "__main__":
   main()   
