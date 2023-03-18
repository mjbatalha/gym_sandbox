__data__ = "17/03/2023"
__version__ = "0.0.1"
__author__ = "Manuel Batalha"

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from pathlib import Path
from tqdm import tqdm

sns.set_theme(style="darkgrid")


class PlotPipeline:

    # todo: summary

    def __init__(self, params: dict):

        # todo: argument description

        # params
        self.pipeline_params = params

        # paths
        self.plot_dir = os.path.join(os.getcwd(), "gym_sandbox", "plots")
        self.model_dir = os.path.join(os.getcwd(), "gym_sandbox", "models")

        # objects
        self.run_names = []
        for run_name in os.listdir(self.model_dir):
            n_run, env, model, alg, reward = run_name.split("__")
            a = int(n_run) in params["n_runs"] or "all" in params["n_runs"]
            b = env in params["envs"] or "all" in params["n_runs"]
            c = model in params["models"] or "all" in params["n_runs"]
            d = alg in params["algorithms"] or "all" in params["n_runs"]
            if a and b and c and d:
                self.run_names.append(run_name)

    def ep_rollmean(self, rs, eps, w):
        rs, eps = np.asarray(rs), np.asarray(eps)
        rm_rs = []
        for ep in eps:
            ep_min, ep_max = ep - w / 2, ep + w / 2
            rm_rs.append(np.nanmean(rs[np.argwhere((eps >= ep_min) & (eps <= ep_max)).flatten()]))
        return rm_rs

    def pipeline(self):

        # iterate over pipeline params items
        for k, v in tqdm(self.pipeline_params.items(), total=len(list(self.pipeline_params.items()))):

            # line plot
            if "lineplot" in k:

                # unpack plot params
                n_step, x, y, hue, style = v["n_step"], v["x"], v["y"], v["hue"], v["style"]
                ep_min, ep_max = v["ep_min"], v["ep_max"]
                rollmean, rollwindow = v["rolling_mean"], v["rolling_window"]

                # iterate over training runs
                df_list = []
                for run_name in self.run_names:

                    # load stats dict
                    stats_path = os.path.join(self.model_dir, run_name, "stats.json")
                    with Path(stats_path).open('r') as f:
                        stats = json.load(f)

                    # apply rolling mean
                    if rollmean:
                        stats[y + "_rm"] = self.ep_rollmean(stats[y], stats["episode"], rollwindow)

                    # apply min & max episodes
                    if isinstance(ep_min, int) and isinstance(ep_max, int):
                        for s, l in stats.items():
                            stats[s] = l[ep_min:ep_max]

                    # add run element names
                    n_run, env, model, alg, reward = run_name.split("__")
                    stats["env"] = [env] * len(stats["episode"])
                    stats["model"] = [model] * len(stats["episode"])
                    stats["train_algorithm"] = [alg] * len(stats["episode"])
                    stats["reward_function"] = [reward] * len(stats["episode"])

                    # append stats dataframe
                    df = pd.DataFrame.from_dict(stats)
                    df_list.append(df)
                df = pd.concat(df_list).reset_index(drop=True)

                # save plot
                save_path = os.path.join(self.plot_dir, self.pipeline_params["plot_run_name"])
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                plt.figure(figsize=(40, 20))
                if not rollmean:
                    sns.lineplot(x=x, y=y, hue=hue, style=style, data=df)
                else:
                    sns.lineplot(x=x, y=y + "_rm", hue=hue, style=style, data=df)
                plt.savefig(os.path.join(save_path, k + ".png"), format='png', dpi=150)

        return True
