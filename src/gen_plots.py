from typing import Callable
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

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
    ignore_columns = ["dataset_size", "EstLInfDiff", "Epsilon", "Random", "Optimal"]
    markers = ["o", "v", "s", "P", "X", "D", "p", "*", "h", "H", "d", "8"]
    dataset_sizes: list = list(set(plotter.df["dataset_size"])) # unique dataset sizes
    dataset_sizes.sort()
    means_across_D_size: dict[str, list[float]] = {}
    cis_across_D_size: dict[str, list[float]] = {}
    # compute the mean and confidence interval for each column we care about
    for dataset_size in dataset_sizes:
        filtered_df = plotter.df[plotter.df["dataset_size"] == dataset_size]
        for column in plotter.df.columns:
            if column not in ignore_columns:
                ci = 1.96 * filtered_df[column].std() / np.sqrt(len(filtered_df))
                cis_across_D_size.setdefault(column, [])
                cis_across_D_size[column].append(ci)
                means_across_D_size.setdefault(column, [])
                means_across_D_size[column].append(filtered_df[column].mean())
    # plot the mean and confidence interval for each column
    for column in plotter.df.columns:
        if column not in ignore_columns:
            ci = cis_across_D_size[column]
            marker = markers.pop()
            plt.errorbar(dataset_sizes, means_across_D_size[column], yerr=ci)
            plt.scatter(dataset_sizes, means_across_D_size[column], label=column, marker=marker)
    # Plot the optimal return and random return as a horizontal line
    plt.axhline(y=plotter.df["Optimal"].mean(), color="black", linestyle="--", label="Optimal")
    plt.axhline(y=plotter.df["Random"].mean(), color="black", linestyle=":", label="Random")

    plt.xlabel("Dataset Size")
    plt.ylabel("Expected Return")
    plt.title(f"Expected Return vs Dataset Size : {plotter.filename}")
    # Move legend to outside of plot
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"plots/returns/{plotter.filename}_returns.pdf")
    plt.clf()

def plot_epsilon_experiment(plotter: Plotter):
    """Plots the experiment returns across the dataset size for the given plotter"""
    ignore_columns = ["dataset_size", "EstLInfDiff", "NBC", "Epsilon", "Random", "Optimal", "True_Epsilon", "LPAL_Rad"]
    markers = ["o", "v", "s", "P", "X", "D", "p", "*", "h", "H", "d", "8"]
    for col in plotter.df.columns:
        if col not in ignore_columns:
            plt.scatter(plotter.df["Epsilon"], plotter.df[col], label=col, marker=markers.pop())
    # Plot vertical lines for lpals rad and the true eps
    plt.axvline(x=plotter.df["True_Epsilon"].mean(), color="black", linestyle="-", label="True Epsilon")
    plt.axvline(x=plotter.df["LPAL_Rad"].mean(), color="black", linestyle="-.", label="LPAL Rad")
    # Plot the optimal return and random return as a horizontal line
    plt.axhline(y=plotter.df["Optimal"].mean(), color="black", linestyle="--", label="Optimal")
    plt.axhline(y=plotter.df["Random"].mean(), color="black", linestyle=":", label="Random")

    plt.xlabel("||Phi^T@u_E - Phi^T@u_E_hat||_inf")
    plt.ylabel("Expected Return")
    plt.title(f"Expected Return vs Epsilon : {plotter.filename}")
    # Move legend to outside of plot
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"plots/epsilon_experiment/{plotter.filename}_eps_experiment.pdf")
    plt.clf()

def plot_return_diffs(plotter: Plotter):
    """Plots rho(u_E) - rho(u_pi) for each method"""
    markers = ["o", "v", "s", "P", "X", "D", "p", "*", "h", "H", "d", "8"]
    ignore_columns = ["dataset_size", "EstLInfDiff", "NBC", "Optimal", "BC", "Random", "Epsilon"]
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

def for_each_dataset(dir: str, fun: Callable):
    """Loop over each dataset in the datasets/ directory and apply fun to it,
    fun must take a Plotter object as its argument"""
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            fname = filename.split(".")[0] # remove the .csv
            print(f"Processing {fname}")
            fun(Plotter(fname, pd.read_csv(f"{dir}/{filename}")))

def main():
    # plot returns
    dir = "datasets"
    for_each_dataset(dir, plot_returns)
    # plot return_diffs
    # for_each_dataset(dir, plot_return_diffs)
    # plot epsilon experiment
    # dir = "datasets/epsilon_experiment"
    # for_each_dataset(dir, plot_epsilon_experiment)

if __name__ == "__main__":
    main()

