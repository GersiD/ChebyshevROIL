from typing import Callable
import matplotlib.pyplot as plt
import os
import pandas as pd

"""This document is an offshoot of the gridworld document for the uses of Gersi Doko"""
"""Mainly for plotting the results of the experiments stored in the datasetes directory"""

class Plotter(object):
    """Wrapper class that keeps track of a dataset and its info for plotting"""
    def __init__(self, filename: str, df: pd.DataFrame):
        self.filename = filename
        self.df = df
        # Set the font type to TrueType Globally
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        # set the font to be Computer Modern (cmr10 doesnt work so we use serif)
        plt.rcParams["font.family"] = "serif"

def plot_returns(plotter: Plotter):
    """Plots the experiment returns across the dataset size for the given plotter"""
    markers = ["o", "v", "s", "P", "X", "D", "p", "*", "h", "H", "d", "8"]
    ignore_columns = ["dataset_size", "EstLInfDiff", "NBC"]
    dataset_size = plotter.df["dataset_size"]
    for column in plotter.df.columns:
        if column not in ignore_columns:
            plt.plot(dataset_size, plotter.df[column], label=column, marker=markers.pop())
    plt.xlabel("Dataset Size")
    plt.ylabel("Expected Return")
    plt.title(f"Expected Return vs Dataset Size : {plotter.filename}")
    # Move legend to outside of plot
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"plots/returns/{plotter.filename}_returns.pdf")
    plt.clf()

def plot_return_diffs(plotter: Plotter):
    """Plots rho(u_E) - rho(u_pi) for each method"""
    markers = ["o", "v", "s", "P", "X", "D", "p", "*", "h", "H", "d", "8"]
    ignore_columns = ["dataset_size", "EstLInfDiff", "NBC", "Optimal", "BC", "Random"]
    x_axis = plotter.df["EstLInfDiff"]
    for column in plotter.df.columns:
        if column not in ignore_columns:
            plt.scatter(x_axis, plotter.df["Optimal"] - plotter.df[column], label=column, marker=markers.pop())
    plt.xlabel("||u_E - u_E_hat||_inf")
    plt.ylabel("rho(u_E) - rho(u_pi)")
    plt.title(f"Regret vs Epsilon : {plotter.filename}")
    # Move legend to outside of plot
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"plots/return_diffs/{plotter.filename}_return_diffs.pdf")
    plt.clf()

def for_each_dataset(fun: Callable):
    """Loop over each dataset in the datasets/ directory and apply fun to it,
    fun must take a Plotter object as its argument"""
    for filename in os.listdir("datasets/"):
        if filename.endswith(".csv"):
            fname = filename.split(".")[0] # remove the .csv
            print(f"Processing {fname}")
            fun(Plotter(fname, pd.read_csv(f"datasets/{filename}")))

def main():
    # plot returns
    # for_each_dataset(plot_returns)
    # plot return_diffs
    for_each_dataset(plot_return_diffs)

if __name__ == "__main__":
    main()

