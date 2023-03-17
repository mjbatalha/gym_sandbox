"""
Entry point to gym_sandbox repository.
"""

__date__ = "17/03/2023"
__version__ = "0.0.1"
__author__ = "Manuel Batalha"

import click
import json
import os

from pathlib import Path

CHOICES = ["train", "plot"]


@click.command()
@click.argument("task", nargs=1, type=click.Choice(CHOICES))
def main(task):

    # params path
    params_path = os.path.join(os.getcwd(), "gym_sandbox", "params")

    if task == "train":

        from .train_pipeline import TrainPipeline

        # load relevant parameters
        model_params = json.load(Path(os.path.join(params_path, "model_params.json")).open('r'))
        train_algorithm_params = json.load(Path(os.path.join(params_path, "train_algorithm_params.json")).open('r'))
        train_pipeline_params = json.load(Path(os.path.join(params_path, "train_pipeline_params.json")).open('r'))

        # nested params
        train_pipeline_params["model_params"] = model_params
        train_pipeline_params["train_algorithm_params"] = train_algorithm_params

        # run pipeline
        p = TrainPipeline(train_pipeline_params)
        success = p.pipeline()
        print(f"Training Pipeline Succeeded: {success}")

    elif task == "plot":

        from .plot_pipeline import PlotPipeline

        # load relevant parameters
        plot_params = json.load(Path(os.path.join(params_path, "plot_params.json")).open('r'))

        # run pipeline
        p = PlotPipeline(plot_params)
        success = p.pipeline()
        print(f"Plot Pipeline Succeeded: {success}")


if __name__ == "__main__":
    main()
