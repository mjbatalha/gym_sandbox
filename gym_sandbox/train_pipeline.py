__data__ = "17/03/2023"
__version__ = "0.0.1"
__author__ = "Manuel Batalha"

import gymnasium as gym
import json
import numpy as np
import os
import shutil
import torch

from pathlib import Path
from tqdm import tqdm

from .models import SeqLinear
from .train_algorithms import DQL, REINFORCE

MODELS = {
    "seqlinear": SeqLinear,
}

ALGORITHMS = {
    "dql": DQL,
    "reinforce": REINFORCE,
}


class TrainPipeline:

    # todo: summary

    def __init__(self, params: dict):

        # todo: argument description

        # params
        self.n_eps = params["n_eps"]
        self.seeds = params["seeds"]
        self.verbose = params["verbose"]
        self.record = params["record"]

        # objects
        self.pipeline_params = params
        self.model_params = params["model_params"][params["model_name"]]
        self.algorithm_params = params["train_algorithm_params"][params["train_algorithm_name"]]
        self.stats = {"episode": [], "reward": [], "episode_length": [], "episode_time": []}

        # paths
        self.model_dir = os.path.join(os.getcwd(), "gym_sandbox", "models")
        self.tmp_dir = os.path.join(os.getcwd(), "tmp")
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        # gym environment
        self.env = gym.make(params["env_name"], render_mode="rgb_array")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, deque_size=params["deque_size"])
        if self.record:
            def episode_trigger(ep):  # todo: add to "record_triggers.py"
                a = ep % params["record_ep_trigger"] >= params["record_ep_trigger"] - params["rec_last_n_eps"]
                b = ep % params["record_ep_trigger"] == 0
                if a or b:
                    return True
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=self.tmp_dir, episode_trigger=episode_trigger,
                                                name_prefix=params["env_name"], disable_logger=True)

    def pipeline(self):

        # iterate over seeds
        for seed in self.seeds:

            # seed & reset environment
            self.env.reset(seed=seed)

            # instantiate model
            model = MODELS[self.pipeline_params["model_name"]](self.model_params)

            # instantiate train algorithm
            alg = ALGORITHMS[self.pipeline_params["train_algorithm_name"]](self.env, model, self.algorithm_params)

            # setup trained model dir
            dir_name = "__".join([str(len(os.listdir(self.model_dir)) + 1).zfill(4),
                                  self.pipeline_params["env_name"],
                                  self.pipeline_params["model_name"],
                                  self.pipeline_params["train_algorithm_name"],
                                  self.algorithm_params["reward"]])
            dir_path = os.path.join(self.model_dir, dir_name)
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            # setup video dir
            if self.record:
                video_dir = os.path.join(dir_path, "videos")
                if not os.path.isdir(video_dir):
                    os.mkdir(video_dir)
                self.env.video_folder = video_dir
                self.env.episode_id = 1

            # reset stats
            self.stats = {"episode": [], "reward": [], "episode_length": [], "episode_time": []}

            # iterate over episodes
            for ep in tqdm(range(1, self.n_eps + 1), total=self.n_eps):

                # episode training step
                alg.train()

                # get episode stats
                self.stats["episode"].append(ep)
                self.stats["reward"].append(float(alg.info["episode"]["r"]))
                self.stats["episode_length"].append(float(alg.info["episode"]["l"]))
                self.stats["episode_time"].append(float(alg.info["episode"]["t"]))

                # verbose
                if self.verbose and ep % self.pipeline_params["deque_size"] == 0:
                    mean_r = int(np.mean(self.env.return_queue))
                    print("Episode:", ep, "Average Reward:", mean_r)

            # save stats
            stats_path = os.path.join(dir_path, "stats.json")
            with Path(stats_path).open('w') as f:
                json.dump(self.stats, f, sort_keys=True, indent=4, separators=(',', ':'))

            # save params
            self.pipeline_params["seed"] = seed
            params_path = os.path.join(dir_path, "training_run_params.json")
            with Path(params_path).open('w') as f:
                json.dump(self.pipeline_params, f, sort_keys=True, indent=4, separators=(',', ':'))

            # save model
            model_path = os.path.join(dir_path, "model.pth")
            torch.save(alg.model.state_dict(), model_path)

        # reset & close environment
        self.env.reset()
        self.env.close()

        # delete tmp dir
        shutil.rmtree(self.tmp_dir)

        return True
