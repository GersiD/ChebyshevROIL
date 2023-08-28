from rmdp import *
import numpy as np

class ThreeStateEnv(MDP):
    """3 State environment from Syed and Schapire 2008, this is supposed to an enviornment where
    LPAL perfoms better than chebyshev center."""

    def __init__(self, gamma:float) -> None:
        self.num_states = 3
        self.num_actions = 2
        # phi has two columns because im lazy and dont want to change the chebyshev code :/
        self.phi = np.array([[0,1,-1,0,1,-1], [0,0,0,0,0,0]]).T
        self.num_features = 1 # this is not a typo, the second column is a dummy column
        self.P = self.compute_transition()
        self.reward = np.array([0,1,-1,0,1,-1]) # this is iterating by action first for some reason
        self.p_0 = np.array([1,0,0]) # start in state 0
        self.gamma = gamma

        # Construct the underlying MDP
        super().__init__(
            self.num_states,
            self.num_actions,
            self.num_features,
            self.P,
            self.phi,
            self.p_0,
            self.gamma,
            self.reward,
        )

    def compute_transition(self) -> np.ndarray:
        S = self.num_states
        A = self.num_actions
        P = np.zeros((S, S, A))
        # P[current_state, next_state, action]
        # state 0 either goes to state 1 or 2 depending on the action
        P[0, 1, 0] = 1
        P[0, 2, 1] = 1
        # state 1 always transitions to itself
        P[1, 1, 0] = 1
        P[1, 1, 1] = 1
        # state 2 always transitions to itself
        P[2, 2, 0] = 1
        P[2, 2, 1] = 1

        for s in range(S):  # ensure P is a transition probablity matrix
            for a in range(A):
                assert sum(P[s, :, a]) == 1
        return P

def main():
    env = ThreeStateEnv(0.99)
    # D = [[(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0)]] # only show state 1 because this is the ideal state
    D = [[(2,0), (2,0),(2,0),(2,0),(2,0),(2,0),(2,0),(2,0),(2,0)]] # only show state 2 even though this is not the ideal state
    # D = [[(2,0)]]
    # D = [[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]] # show what to do in the starting state
    # D = [[(0,0),(1,0),(2,0),(0,0),(0,0),(0,0),(0,0),(1,0),(1,0)]] # show what to do in all states
    print(f"\033[32mOptimal Return = { env.opt_return }\033[0m")
    # print(f"\033[32mOptimal occ_freq = { env.u_E }\033[0m")
    # print(f"\033[32mOptimal policy = { env.occupancy_freq_to_policy(env.u_E) }\033[0m")
    (u_cheb, radius, cheb_return) = env.solve_chebyshev_center(D)
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
