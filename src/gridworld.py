import numpy as np
from numpy import size
from rmdp import *


class GridWorld(MDP):
    """
    Implementation of a GriDWorld MDP for testing out RL methods.
    Here we assume k-colored features and rewards. For sake of simplicity k = 4.

    num_rows = the number of rows and cols in the gridworld game
    gamma = discount factor
    reward = optional SA reward vector
    """

    def __init__(self, num_rows: int, gamma, phi=None, reward=None, p_0=None):
        # Set self values
        self.num_rows = num_rows
        self.num_states = num_rows * num_rows
        self.num_actions = 4  # Up, down, left, right

        # Set the number of colors for the gridworld
        # Later this could be passed in but for now its simpler to just say 4 colored graph
        self.k = 4

        # Compute features and transition
        self.P = self.compute_transition()
        if phi is None:
            self.phi = self.compute_features()  # SA by K matrix iterating by action first
        else:
            self.phi = phi
        self.num_features = size(self.phi, 1)  # Number of columns

        self.gamma = gamma
        self.reward = reward  # reward should be an SA vector iterating by action first
        if reward is None:  # if reward is none
            self.reward = self.compute_reward()
        if p_0 is None:
            self.p_0 = 1 / self.num_states * np.ones(self.num_states)
        else:
            self.p_0 = p_0
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

    def compute_features(self) -> np.ndarray:
        """Compute the feature matrix, for now it returns a SA X K matrix where each column represents a color of an sa cell
        Note this is where the color of each state is decided. Note phi iterates by state first.
        """
        k = self.k  # number of colors
        phi_s = np.zeros((self.num_states, k))
        # pick a color for each state
        indices = np.random.randint(0, k, size=self.num_states)
        for state in range(self.num_states):
            phi_s[state, indices[state]] = 1
        phi = np.vstack([phi_s for _ in range(self.num_actions)]) * 100

        if np.linalg.matrix_rank(phi) == self.k:  # Ensure phi is of full rank
            print("Phi is of full rank!")
            return phi
        else:  # If its not try again I guess
            return self.compute_features()

    def compute_reward(self) -> np.ndarray:
        """Compute reward vector which is realizeable by phi which is computed in GridWorld.compute_features
           Returns a SA vector iterating by state first"""
        # use k-colored rewards k set in constructor
        # Ensure these rewards are realizeable by phi
        k = self.k
        # Sample from LInf Ball and reject if L1 norm is greater than 1
        # Ensure the L1 norm of w <= 1 where phi.T w = r
        w = np.random.rand(k) * 2 - 1
        while np.linalg.norm(w, ord=1) > 1:
            w = np.random.rand(k) * 2 - 1

        # matrix of size S x K which shows the color of each state
        phi_s = self.phi[range(self.num_states)]
        rewards_s = phi_s @ w
        rewards = np.hstack([rewards_s for _ in range(self.num_actions)])
        print(f"\033[31mNorm of w = { np.linalg.norm(w,ord=1) }\033[0m")

        return rewards

    def compute_transition(self) -> np.ndarray:
        """Compute the transition matrix P, the default matrix ensures equal probablity of transitioning to any orthogonal state
        P is indexed by [cur_state, next_state, action]"""
        S = self.num_states
        A = self.num_actions
        num_rows = self.num_rows
        P = np.zeros((S, S, A))
        p1, p2 = 0.2, 0.2  # .4 in the direction you want to go and .2 in the other directions
        # p1, p2 = 0.25, 0.0  # Fully Random
        # p1, p2 = 0.0, 1.0 # Deterministic
        assert 4 * p1 + p2 == 1.0
        action_prob = p1 * np.ones((4, 4)) + p2 * np.eye(4)

        for state in range(S):
            for action in range(A):
                # if action == 0:
                # the agent goes to the right
                if (state + 1) % num_rows != 0:
                    P[state, state + 1, action] += action_prob[action, 0]
                else:
                    P[state, state, action] += action_prob[action, 0]
                # the agent goes up
                if state >= num_rows:
                    P[state, state - num_rows, action] += action_prob[action, 1]
                else:
                    P[state, state, action] += action_prob[action, 1]
                # the agent goes to the left
                if (state + 1) % num_rows != 1:
                    P[state, state - 1, action] += action_prob[action, 2]
                else:
                    P[state, state, action] += action_prob[action, 2]
                # the agent goes down
                if state + num_rows < S:
                    P[state, state + num_rows, action] += action_prob[action, 3]
                else:
                    P[state, state, action] += action_prob[action, 3]

        # for s in range(S):  # ensure P is a transition probablity matrix
        #     for a in range(A):
        #         assert sum(P[s, :, a]) - 1 < 1e-10

        return P


def main():
    np.random.seed(603)
    env = GridWorld(3, 0.99)
    episodes = 1
    horizon = 100
    D = env.generate_demonstrations_from_occ_freq(env.u_E, episodes, horizon)
    # D = env.generate_off_policy_demonstrations(episodes, horizon, env.u_E, env.u_rand)
    # D = [env.generate_all_expert_demonstrations()]
    # print(env.reward)
    bc_return = env.solve_BC(D, episodes, horizon)
    print(f"\033[34mBC Return    = { bc_return }\033[0m")
    gail_ret = env.solve_GAIL(D, episodes, horizon)
    print(f"GAIL Return    = { gail_ret }")
    print(f"\033[32mOptimal Return = { env.opt_return }\033[0m")
    (_, radius, cheb_return) = env.solve_chebyshev_center(D)
    print(f"\033[34mCheb Return    = { cheb_return }\033[0m")
    print(f"\033[34mCheb Radius    = { radius }\033[0m")
    (_, radius, syed_return) = env.solve_syed(D, episodes, horizon)
    print(f"Syed Return    = { syed_return }")
    print(f"Syed Radius    = { radius }")
    print(f"Random return  = { env.random_return }")

    # print(env.occupancy_freq_to_policy(cheb_sol[0])) # print the policy learned


if __name__ == "__main__":
    main()
