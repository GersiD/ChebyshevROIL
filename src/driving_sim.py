from types import LambdaType
from rmdp import MDP
from typing import List, Tuple
import numpy as np

class DrivingSim(MDP):
    def __init__(self, num_rows: int, obstacles: List[int] = []):
        if (num_rows < 3):
            raise ValueError("Number of rows must be greater than or equal to 3")
        self.num_rows = num_rows
        self.num_states = num_rows * num_rows
        self.num_actions = 3
        self.obstacles = obstacles
        P = self.compute_transition()
        phi, reward = self.compute_features_and_rewards()
        # we want it to start on the bottom row but not on the edges
        p_0 = np.zeros(self.num_states) 
        p_0[(self.num_rows - 1) * self.num_rows + 1 : (self.num_rows - 1) * self.num_rows + self.num_rows - 1] = 1 / (self.num_rows - 2)
        assert(sum(p_0) - 1 < 1e-10)
        super().__init__(self.num_states, self.num_actions, 4, P, phi, p_0, 0.99, reward)

    def compute_transition(self) -> np.ndarray:
        """
        Compute the transition matrix,
        The actor can either stay in the same lane, move left, or move right
        staying progresses them to the row about them with some noise and wraps around the board
        moving left or right moves them to the corresponding column above them with some noise and wraps around the board
        """
        S = self.num_states
        A = self.num_actions
        P = np.zeros((S, S, A))
        STAY = 0
        LEFT = 1
        RIGHT = 2
        noise = 0.1
        fwd = lambda s: (s - self.num_rows) % S
        left = lambda s: (s - (self.num_rows + 1)) % S
        right = lambda s: (s - (self.num_rows - 1)) % S
        # stay mechanics
        for s in range(S):
            col = s % self.num_rows
            # left border
            if col == 0:
                P[s, fwd(s), STAY] += 1.0 - noise
                P[s, right(s), STAY] += noise
            # right border
            elif col == self.num_rows - 1:
                P[s, fwd(s), STAY] += 1.0 - noise
                P[s, left(s), STAY] += noise
            # middle
            else:
                P[s, fwd(s), STAY] += 1.0 - (noise * 2)
                P[s, left(s), STAY] += noise
                P[s, right(s), STAY] += noise
        # left mechanics
        for s in range(S):
            col = s % self.num_rows
            # left border
            if col == 0:
                P[s, fwd(s), LEFT] += 1.0 - noise
                P[s, right(s), LEFT] += noise
            # right border
            elif col == self.num_rows - 1:
                P[s, left(s), LEFT] += 1.0 - noise
                P[s, fwd(s), LEFT] += noise
            # middle
            else:
                P[s, left(s), LEFT] += 1.0 - (noise * 2)
                P[s, fwd(s), LEFT] += noise
                P[s, right(s), LEFT] += noise
        # right mechanics
        for s in range(S):
            col = s % self.num_rows
            # left border
            if col == 0:
                P[s, right(s), RIGHT] += 1.0 - noise
                P[s, fwd(s), RIGHT] += noise
            # right border
            elif col == self.num_rows - 1:
                P[s, fwd(s), RIGHT] += 1.0 - noise
                P[s, left(s), RIGHT] += noise
            # middle
            else:
                P[s, right(s), RIGHT] += 1.0 - (noise * 2)
                P[s, fwd(s), RIGHT] += noise
                P[s, left(s), RIGHT] += noise
        for s in range(S):  # ensure P is a transition probablity matrix
            for a in range(A):
                assert(sum(P[s, :, a]) - 1 < 1e-10)
        return P

    def compute_features_and_rewards(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the feature matrix, phi, which is a matrix of size SA x K which shows what features are active for each state-action pair
        There are 4 features:
        0 - The row the car is in
        1 - The column the car is in
        2 - Did the car crash
        3 - Did the car hit the bumpers
        Also Computes Rewards which are a vector of size SA which shows the reward for each state-action pair
        computed with a linear function of the features with weights w = [0.1, 0.1, 0.4, 0.2] which is in the L1 ball of radius 1

        returns phi and rewards
        """
        S = self.num_states
        A = self.num_actions
        bumper_penalty = -100
        crash_penalty = -1000
        phi = np.zeros((S, 4))
        for s in range(S):
            row = s // self.num_rows
            col = s % self.num_rows
            phi[s, 0] = row
            phi[s, 1] = col
            phi[s, 3] = bumper_penalty if col == 0 or col == self.num_rows - 1 else 0
        for s in self.obstacles:
            phi[s, 2] = crash_penalty
        phi_SA = np.vstack([phi for _ in range(A)])
        weights = np.array([0.1, 0.1, 0.4, 0.2])
        return phi_SA, phi_SA @ weights

    def display(self):
        """Display the current state of the board"""
        board = np.zeros((self.num_rows, self.num_rows))
        print("Reward View")
        for s in range(self.num_states):
            row = s // self.num_rows
            col = s % self.num_rows
            board[row, col] = np.sum(self.reward_matrix[s, :])
        print(board)
        print("Obstacle View")
        for s in range(self.num_states):
            row = s // self.num_rows
            col = s % self.num_rows
            board[row, col] = 1 if s in self.obstacles else 0
        print(board)
        print("Opt Policy View")
        for s in range(self.num_states):
            row = s // self.num_rows
            col = s % self.num_rows
            board[row, col] = np.argmax(self.opt_policy[s])
        print(board)

if __name__ == "__main__":
    num_rows = 10
    # np.random.see(603)
    obstacles = list(np.random.choice((num_rows*num_rows) - (num_rows),  num_rows, replace=False))
    dsim = DrivingSim(num_rows, obstacles)
    # dsim.display()
    
    env = DrivingSim(5)
    episodes = 1
    horizon = 10
    D = env.generate_demonstrations_from_occ_freq(env.u_E, episodes, horizon)
    print(f"\033[32mOptimal Return = { env.opt_return }\033[0m")
    (_, radius, cheb_return) = env.solve_chebyshev_center(D)
    print(f"\033[34mCheb Return    = { cheb_return }\033[0m")
    print(f"\033[34mCheb Radius    = { radius }\033[0m")
    (eps, _, rad_2, cheb_return_2) = env.solve_cheb_part_2(D, False)
    print(f"\033[34mCheb Return2    = { cheb_return_2 }\033[0m")
    print(f"\033[34mCheb Radius2    = { rad_2 }\033[0m")
    (_, radius, syed_return) = env.solve_syed(D, episodes, horizon)
    print(f"Syed Return    = { syed_return }")
    print(f"Syed Radius    = { radius }")
    print(f"Random return  = { env.random_return }")

