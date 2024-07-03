"""
Merger script the results of different runs into a single csv file
"""
from pathlib import Path
from shutil import rmtree

import click
import pandas as pd

import mvgl

@click.command()
@click.option("--experiment")
@click.option("--method")
@click.option("--n-runs", type=int)
def main(
    experiment, method, n_runs
):
    pass

    # Read the results of the individual runs
    performances = []
    save_path = Path(mvgl.ROOT_DIR, "data", "simulations", "outputs", experiment, method)
    
    for seed in range(n_runs):
        save_file = Path(save_path, f"run-{seed}.csv")

        perf = pd.read_csv(save_file, index_col=0)
        performances.append(perf)

    rmtree(save_path)

    # Combine the results into a single dataframe
    performances = pd.concat(performances, ignore_index=True)

    save_file = Path(mvgl.ROOT_DIR, "data", "simulations", "outputs", experiment,
                     f"{method}.csv")
    performances.to_csv(save_file)

if __name__ == "__main__":
    main()