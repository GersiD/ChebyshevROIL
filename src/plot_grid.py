from matplotlib.cbook import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gridworld import GridWorld
from typing import List, Tuple
from enum import Enum

from lava_world import LavaWorld

class Action(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class PlotGrid:
    def __init__(self, num_rows: int, gamma: float):
        self.num_rows = num_rows
        self.gamma = gamma
        # self.grid_world = GridWorld(num_rows, gamma)
        self.grid_world = LavaWorld(0.99)
        self.num_states = self.grid_world.num_states
        self.num_actions = self.grid_world.num_actions
        self.rewards = self.grid_world.reward[0:self.num_states].reshape(num_rows, num_rows)
        self.cmap = matplotlib.colormaps["viridis"]
        self.cmap.set_under("black")
        self.cmap.set_over("white")
        self.P = self.grid_world.P
        self.D = self.grid_world.generate_demonstrations_from_occ_freq(self.grid_world.u_E, num_rows, num_rows)

    def gen_x_direction(self, action: int) -> int:
        if action == 0:  # right
            return 1
        elif action == 1 or action == 3:  # up or down respectively
            return 0
        elif action == 2:  # left
            return -1
        else:
            raise ValueError(f"Invalid action {action}.")

    def gen_y_direction(self, action: int) -> int:
        if action == 0 or action == 2:  # right or left respectively
            return 0
        elif action == 1:  # up
            return 1
        elif action == 3:  # down
            return -1
        else:
            raise ValueError(f"Invalid action {action}.")

    def gen_U_row(self, policy_row: np.ndarray) -> np.ndarray:
        U_row = np.zeros(len(policy_row))
        for action, prob in enumerate(policy_row):
            U_row[action] = prob * self.gen_x_direction(action)
        return U_row

    def gen_V_row(self, policy_row: np.ndarray) -> np.ndarray:
        V_row = np.zeros(len(policy_row))
        for action, prob in enumerate(policy_row):
            V_row[action] = prob * self.gen_y_direction(action)
        return V_row

    def on_click(self, event):
        if event.button is plt.MouseButton.LEFT and event.inaxes is not None:
            x,y = int(round(event.xdata)), int(round(event.ydata))
            cur_cell = x + y*self.num_rows
            print(f"You clicked on state {cur_cell} at cell ({x},{y})")
            print(f"R({cur_cell}) = {self.rewards[y,x]}")
            print(f"Probabilies for state {cur_cell} are ...")
            for action, col in enumerate(self.P[cur_cell,:,:].T):
                action_name = Action(action) # Action enum to keep track of what action is what
                possible_next_states = np.nonzero(col)[0]
                print(f"{action_name} ({action})")
                for next_state in possible_next_states:
                    print(f"\t\tS' = {next_state} with probability {col[next_state]}")
            print("\n")
            # ax = event.inaxes
            # ax.scatter(x,y,marker="o",color="b",s=100)
            # ax.figure.canvas.draw()

    def plot_grid(self):
        im = plt.matshow(self.rewards, interpolation="nearest", cmap=self.cmap)
        cbar = plt.colorbar(im)
        cbar.set_label("Reward (Higher is better)", rotation=270, labelpad=20)

        plt.gca().set_xticks(np.arange(0.5, self.num_rows, 1), minor=True)
        plt.gca().set_yticks(np.arange(0.5, self.num_rows, 1), minor=True)
        plt.gca().grid(which="minor", color="w", linestyle="-", linewidth=2)
        plt.gca().tick_params(which="minor", bottom=False, left=False)

    def plot_policy(self, policy):
        X, Y = np.meshgrid(np.arange(0, self.num_rows), np.arange(0, num_rows))
        X = np.tile(X, (1, self.num_actions))
        X = np.reshape(X.T, (self.num_states, self.num_actions), order="F")
        Y = np.tile(Y, (1, self.num_actions))
        Y = np.reshape(Y, (self.num_states, self.num_actions))
        U = np.apply_along_axis(self.gen_U_row, 1, policy)
        V = np.apply_along_axis(self.gen_V_row, 1, policy)
        plt.quiver(X, Y, U, V, scale=1.0, units="xy", color="black", width=0.03)

    def plot(self, policy):
        self.plot_grid()
        self.plot_policy(policy)
        plt.connect('button_press_event', self.on_click)
        plt.show()

    def update(self, frame):
        pass

    def timelapse(self, policy, delta_D_size):
        animation = matplotlib.animation.FuncAnimation(plt.gcf(), self.update, frames=range(0, self.num_states**2, delta_D_size), interval=1, repeat=False)
        animation.save('timelapse.gif')

        
# To whoever cares to read this, please know this took much too long to figure out
# I'm sorry for the lack of comments
if __name__ == "__main__":
    num_rows = 5
    np.random.seed(42)
    plot = PlotGrid(num_rows, 0.99)
    env = plot.grid_world

    episodes = 1
    horizon = 1
    np.random.seed(42)
    D = env.generate_demonstrations_from_occ_freq(env.u_E, episodes, horizon)
    # D: List[List[Tuple[int,int]]]= [[(10,0), (11,0), (12,0), (13,0), (14, 3), (19,3), (24, 2), (23,2), (22,2), (21,2), (20,3)]]
    # D: List[List[Tuple[int,int]]]= [[(10,0), (11,0), (12,0), (13,0), (14, 3), (19,3), (24, 2), (23,2), (22,2), (21,2)]]
    # D: List[List[Tuple[int,int]]]= [[(10,0), (21, 2), (14,3)]]
    horizon = len(D[0])
    cheb_occ, cheb_rad, cheb_return = env.solve_chebyshev_center(D)
    syed_occ, syed_rad, syed_return = env.solve_syed(D, episodes, horizon)
    print(f"Optimal: {env.opt_return}")
    print(f"Chebyshev: {cheb_return}")
    print(f"Cheb_rad: {cheb_rad}")
    print(f"Syed: {syed_return}")
    print(f"Random: {env.random_return}")
    # policy = env.occupancy_freq_to_policy(env.u_E)
    policy = env.occupancy_freq_to_policy(cheb_occ)
    # policy = env.occupancy_freq_to_policy(syed_occ)
    # policy = np.ones((env.num_states, env.num_actions)) / env.num_actions
    # print(policy)
    plot.plot(policy)
