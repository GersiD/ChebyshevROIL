from typing import List, Set, Tuple
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor as Pool
import gurobipy as gp
from gurobipy import GRB


class MDP(object):
    """MDP class for use in the following methods which solve the MDP"""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_features: int,
        P,
        phi,
        p_0,
        gamma,
        reward=None,
    ):
        # super(MDP, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.states = np.arange(self.num_states)
        self.actions = np.arange(self.num_actions)
        self.num_features = num_features
        self.gamma = gamma  # discount factor
        # transition probability P \in S x S x A
        self.P = P
        # initial state distribution p_0 \in S
        self.p_0 = p_0
        # reward(S x A) vector itrating by state first
        self.reward = reward
        self.reward_matrix = np.reshape(
            self.reward, (self.num_states, self.num_actions), order="F"
        )
        # features matrix phi \in (SxA)xK where K is the number of features
        self.phi = phi
        self.phi_matrix = phi.reshape(self.num_states, self.num_actions, self.num_features, order='F')
        # Stacked (I - gamma P_a)
        self.IGammaPAStacked = self.construct_design_matrix()
        # occupancy frequency of an expert's policy u[S x A]
        (u_E, opt_return) = self.solve_putterman_dual_LP_for_Opt_policy()
        self.u_E = u_E  # occupancy frequency of the expert's policy
        self.opt_return = opt_return  # optimal return of the expert's policy
        self.random_return = (
            self.generate_random_policy_return()
        )  # return of a random policy
        # feature expectation of an expert's policy mu_[K]
        self.mu_E = None
        self.weights = np.zeros(num_features)

    def next_state(self, state: int, action: int) -> int:
        """Given state and action pair, return the next state based on the MDP dynamics"""
        return np.random.choice(self.states, p=self.P[state, :, action])

    def occupancy_freq_to_policy(self, u):
        """Converts u which is an occupancy frequency matrix of size S x A to a policy of size S x A"""
        S = self.num_states
        A = self.num_actions
        policy = np.zeros((S, A))
        sum_u_s = np.sum(u, axis=1)
        for s in range(S):
            policy[s, :] = u[s, :] / max(sum_u_s[s], 0.0000001)
        return policy

    def generate_samples_from_policy(
        self, num_samples, policy
    ) -> List[Tuple[int, int]]:
        """Generate samples from the given policy
        policy = policy should be an SxA matrix where each row sums to 1
        """
        D = []  # Dataset of (s, a) pairs
        cur_state = np.random.choice(self.states, p=self.p_0)
        for _ in range(num_samples):
            action = np.random.choice(self.actions, p=policy[cur_state, :])
            D.append((cur_state, action))
            cur_state = self.next_state(cur_state, action)
        return D

    def generate_random_policy_return(self) -> float:
        """Generate the return of a uniformly random policy where pi(a|s) = 1/|A|"""
        # Calculate P_pi for randomized pi
        P_pi = np.sum(self.P, axis=2) / self.num_actions
        r_pi = np.sum(self.reward_matrix, axis=1) / self.num_actions
        d_pi = np.linalg.inv(np.eye(self.num_states) - self.gamma * P_pi.T) @ self.p_0
        return d_pi @ r_pi

    def generate_samples_from_occ_freq(
        self, num_samples, occupancy_freq
    ) -> List[Tuple[int, int]]:
        """
        Generate samples from the given occupancy frequency
        occupancy_freq = matrix of size S x A
        num_samples = number of samples to collect
        """
        return self.generate_samples_from_policy(
            num_samples, self.occupancy_freq_to_policy(occupancy_freq)
        )

    def generate_expert_demonstrations(self, num_samples) -> List[Tuple[int, int]]:
        """A wrapper around generate_samples which calls it with the optimal occupancy_freq calculated by
        the dual putterman solution"""
        return self.generate_samples_from_occ_freq(num_samples, self.u_E)

    def generate_all_expert_demonstrations(self) -> List[Tuple[int, int]]:
        """Returns a list of all s,a pairs that the expert follows"""
        policy = self.occupancy_freq_to_policy(self.u_E)
        D = []
        for s in self.states:
            a = np.random.choice(self.actions, p=policy[s, :])
            D.append((s, a))
        return D

    class SampleCollector:
        """This class is soley for the pickling required by the multiprocessing pool
        it remembers the occ_freq for the following call to generate_samples"""

        def __init__(self, mdp, policy, horizon):
            self.policy = policy
            self.mdp = mdp
            self.horizon = horizon

        def __call__(self, _) -> List[Tuple[int, int]]:
            return self.mdp.generate_samples_from_policy(
                self.horizon, policy=self.policy
            )

    def generate_demonstrations_from_occ_freq(
        self, occ_freq, episodes=1, horizon=10, num_samples=None
    ) -> List[List[Tuple[int, int]]]:
        """Generate demonstrations from an occ freq
        Args:
            occ_freq: The occupancy frequency to use
            episodes: The number of episodes to generate
            horizon: The horizon of each episode
            num_samples: The number of samples to generate. If None, then episodes and horizon are used
        Returns:
            A list of episodes, where each episode is a list of (s,a) pairs
        """
        if num_samples:
            return [self.generate_samples_from_occ_freq(num_samples, occ_freq)]
        D: List[List[Tuple[int, int]]] = []
        # gen_samples_closure = self.SampleCollector(
        #     self, self.occupancy_freq_to_policy(occ_freq), horizon
        # )
        # with Pool(1) as pool:
        #     D = list(pool.map(gen_samples_closure, range(episodes)))
        policy = self.occupancy_freq_to_policy(occ_freq)
        for _ in range(0, episodes):
            D.append(self.generate_samples_from_policy(horizon, policy))
        return D

    def construct_design_matrix(self) -> np.ndarray:
        """
        Construct the design matrix consisting of (I - gamma P_a) stacked on top of eachother
        Returns an (SA X S) matrix
        """
        arrays = []
        I = np.eye(self.num_states)
        for action in self.actions:
            arrays.append((I - self.gamma * self.P[:,:,action]))
        return np.vstack(arrays)

    def solve_putterman_dual_LP_for_Opt_policy(self) -> Tuple[np.ndarray, float]:
        """This method solves the problem of Bellman Flow Constraint. This
        problem is sometimes called the dual problem of min p0^T v, which finds
        the optimal value function.

        Returns:
            ndarray: The optimal policy
            float: optimal return
        """
        method = "Dual_LP"
        a = self.num_actions
        s = self.num_states
        p_0 = self.p_0
        gamma = self.gamma
        r = self.reward
        W = self.IGammaPAStacked

        # Model
        model = gp.Model(method)
        model.Params.OutputFlag = 0
        # Variables
        u = model.addMVar(shape=(s * a), lb=0.0)
        # Constraints
        model.addMConstr(W.T, u, "==", p_0)
        # setting the objective
        model.setObjective(r @ u, GRB.MAXIMIZE)
        # Solve
        model.optimize()
        if model.Status != GRB.Status.OPTIMAL:
            raise ValueError("DUAL LP DID NOT FIND OPTIMAL SOLUTION")
        dual_return = model.objVal

        u_flat = u.X  # Gurobi way of getting the value of a model variable
        # Check to make sure u is an occupancy frequency
        # assert np.sum(u_flat) - 1 / (1 - gamma) < 10**-2
        # Return the occ_freq and the opt return
        return u_flat.reshape((s, a), order="F"), dual_return

    def observed(self, state, D: Set[Tuple]) -> Tuple[bool, int]:
        for s, a in D:
            if state == s:
                return (True, a)
        return (False, -1)

    def construct_constraint_vector(self, D: Set[Tuple]) -> np.ndarray:
        """Constructs the constraint vector for u in Upsilon
        Returns a vector of length num_states * num_actions"""
        c = np.zeros((self.num_states, self.num_actions))
        for state in self.states:
            (observed_state, observed_action) = self.observed(state, D)
            for action in self.actions:
                # In order to be in Upsilon, you must observe the state with that action
                # Consistent with the expert
                if observed_state and action != observed_action:
                    # a constraint of 1 means that you shouldnt choose that (s,a) pair
                    c[state, action] = 1
        return c.reshape((self.num_states * self.num_actions), order="F")

    def solve_chebyshev_center(
        self, D: List[List[Tuple[int, int]]]
    ) -> Tuple[np.ndarray, float, float]:
        """Solves the chebyshev center problem to find the optimal occupancy_freq, more details in the paper
        Returns an SA vector u in U, the chebyshev radius, and the optimal return"""
        method = "Chebyshev"
        s = self.num_states
        a = self.num_actions
        phi = self.phi
        p_0 = self.p_0
        D_flat = set(itertools.chain.from_iterable(D))
        # print(
        #     f"len(D_flat) = {len(D_flat)} len(D) = {len(list(itertools.chain.from_iterable(D)))}"
        # )
        c = self.construct_constraint_vector(D_flat)
        W = self.IGammaPAStacked  # (I-\gamma P_a) stacked vertically

        model = gp.Model(method)
        model.Params.OutputFlag = 0

        # Define model variables
        sigma = model.addVar(lb=0.0, obj=1.0)
        u = model.addMVar(shape=(s * a), lb=0.0)
        alpha = model.addMVar(
            shape=(self.num_features), lb=-GRB.INFINITY, ub=GRB.INFINITY
        )
        alphaHat = model.addMVar(
            shape=(self.num_features),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )
        beta = model.addMVar(
            shape=(s, self.num_features), lb=-GRB.INFINITY, ub=GRB.INFINITY
        )
        betaHat = model.addMVar(
            shape=(s, self.num_features),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
        )

        # Add constraints for features
        for i in range(self.num_features):
            model.addConstr(
                (p_0 @ beta[:, i]) <= sigma + (phi[:, i] @ u)
            )
            model.addConstr(
                (p_0 @ betaHat[:, i]) <= sigma - (phi[:, i] @ u)
            )
            model.addConstr(phi[:, i] <= alpha[i] * c + (W @ beta[:, i]))
            model.addConstr(
                -1 * phi[:, i] <= alphaHat[i] * c + (W @ betaHat[:, i])
            )

        # Add constraints for u \in U
        model.addMConstr(W.T, u, "==", p_0)

        # setting the objective
        model.setObjective(sigma, GRB.MINIMIZE)

        # model.write("./" + method + ".lp") # write the model to a file, for debugging
        # Solve
        model.optimize()
        if model.Status != GRB.Status.OPTIMAL:
            raise ValueError(f"{method} DID NOT FIND OPTIMAL SOLUTION")
        radius = model.objVal
        u_flat = u.X  # Gurobi way of getting the value of a model variable
        # Check to make sure u is an occupancy frequency
        # assert np.sum(u_flat) - 1 / (1 - self.gamma) < 10**-2
        return u_flat.reshape((s, a), order="F"), radius, u_flat.T @ self.reward

    def compute_V_hat(self, D: List[List[Tuple[int, int]]], episodes, horizon):
        phi = self.phi
        gamma = self.gamma
        V = np.zeros(self.num_features)
        for i in range(self.num_features):
            # note phi_i is a matrix of size SxA for indexing purposes
            phi_i = phi[:, i].reshape((self.num_states, self.num_actions), order="F")
            for m in range(episodes):
                for h in range(horizon):
                    s, a = D[m][h]
                    V[i] += phi_i[s, a] * (gamma**h)
            V[i] /= episodes
        return V

    def solve_syed(self, D: List[List[Tuple[int, int]]], episodes: int, horizon: int):
        """Solve the LPAL LP from Syed.
        Args:
            D: list of trajectories (which themselves are lists of length horizon)
            episodes: number of episodes
            horizon: horizon of each episode
        Returns:
            a occupancy_freq matrix, the radius of the LPAL set, and the return achieved.
        """
        method = "Syed LPAL"
        s = self.num_states
        a = self.num_actions
        p_0 = self.p_0
        phi = self.phi
        W = self.IGammaPAStacked
        # Using the experts sample trajectories D, compute an epsilon-good estimate of V
        V_hat = self.compute_V_hat(D, episodes, horizon)
        # Solve the LPAL formulation
        model = gp.Model(method)
        model.Params.OutputFlag = 0

        u = model.addMVar(shape=(self.num_states * self.num_actions), name="u", lb=0.0)
        B = model.addVar(name="B", lb=-GRB.INFINITY)
        for i in range(self.num_features):
            model.addConstr(B <= (phi[:, i] @ u) - V_hat[i])
        model.addMConstr(W.T, u, "==", p_0)

        # model.write("./" + method + ".lp") # write the model to a file, for debugging
        model.setObjective(B, GRB.MAXIMIZE)

        # Solve
        model.optimize()
        if model.Status != GRB.Status.OPTIMAL:
            raise ValueError(f"{method} DID NOT FIND OPTIMAL SOLUTION")

        u_flat = u.X  # Gurobi way of getting the value of a model variable
        radius = model.objVal
        # Check to make sure u is an occupancy frequency
        # assert np.sum(u_flat) - 1 / (1 - self.gamma) < 10**-2
        return u_flat.reshape((s, a), order="F"), radius, u_flat @ self.reward

    def sigmoid_policy(self, theta: np.ndarray, state: int) -> np.ndarray:
        """Compute the sigmoid policy for a given state and parameter vector theta
        theta = parameter vector of size K
        returns a vector of size A"""
        numerator = np.exp(self.phi_matrix[state,:] @ theta)
        return numerator / np.sum(numerator)

    def u_hat(self, state: int, action: int, D:List[List[Tuple[int,int]]]) -> float:
        """Compute the empirical occupancy frequency of a given state-action pair
        returns a float"""
        count = 0
        for d in D:
            for t, (s,a) in enumerate(d):
                if s == state and a == action:
                    count += self.gamma**t
        return count / len(D)

    def D_w(self, state: int, action: int, w: np.ndarray):
        """Compute the sigmoid function for the current state, action
            returns a float"""
        return 1 / (1 + np.exp(-1 * self.phi_matrix[state,action] @ w))

    def generate_samples_from_sigmoid_policy(self, theta: np.ndarray, episodes: int, horizon: int) -> List[List[Tuple[int, int]]]:
        """Generate samples from the sigmoid policy
        theta = parameter vector of size K
        returns a list of episodes, each of length horizon"""
        D: List[List[Tuple[int, int]]] = []
        for _ in range(0, episodes):
            d = []
            cur_state = np.random.choice(self.states, p=self.p_0)
            for _ in range(0, horizon):
                action = np.random.choice(self.actions, p=self.sigmoid_policy(theta, cur_state))
                d.append((cur_state, action))
                cur_state = self.next_state(cur_state, action)
            D.append(d)
        return D

    def solve_GAIL(self, D_e: List[List[Tuple[int, int]]], episodes: int, horizon: int):
        method = "GAIL"
        # TODO make sure that this is correct
        phi = self.phi_matrix

        # Solve the GAIL formulation
        theta_cur = np.random.rand(self.num_features)
        theta_next = np.random.rand(self.num_features)
        w_cur = np.random.rand(self.num_features)

        while np.linalg.norm(theta_cur - theta_next,ord=np.inf) >= 1e-3:  # check loop condition
            # sample from pi_theta
            D_theta = self.generate_samples_from_sigmoid_policy(theta_cur, episodes, horizon)
            # update w
            part_1 = np.zeros(self.num_features)
            for m in range(episodes):
                for h in range(horizon):
                    s, a = D_theta[m][h]
                    part_1 += self.u_hat(s,a,D_theta) * (self.D_w(s,a,w_cur) - 1) * phi[s,a,:]
            part_1 /= episodes
            part_2 = np.zeros(self.num_features)
            for m in range(episodes):
                for h in range(horizon):
                    s, a = D_e[m][h]
                    part_2 += self.u_hat(s,a,D_e) * (self.D_w(s,a,w_cur)) * phi[s,a,:]
            part_2 /= episodes
            w_cur = w_cur + (part_1 + part_2)
            # update theta
            pass
        # Questions
        # What is the gradient of H(pi_theta) wrt theta?

    def solve_MILO(self):
        pass
