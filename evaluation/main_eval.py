# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import os
import sys
import json
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

verl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../verl/verl"))
print("project_root: ", verl_root)
if verl_root not in sys.path:
    sys.path.insert(0, verl_root)

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


@ray.remote
def process_item(config, question_type, metric, response, ground_truth):
    """
    Compute the reward score for a single item using the custom reward function.
    """
    reward_fn = get_custom_reward_fn(config)
    score = reward_fn(metric, response, ground_truth)
    return question_type, metric, score


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    parquet_dir = config.data.path
    
    # -----------------------------
    # Find all Parquet files to evaluate
    # -----------------------------
    parquet_files = [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet")
    ]
    if len(parquet_files) == 0:
        print(f"No parquet files found in {parquet_dir}")
        return
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(**OmegaConf.to_container(config.ray_kwargs.get("ray_init", {})))

    # Store all metric scores
    all_rewards = defaultdict(lambda: defaultdict(list))

    # -----------------------------
    # Process each Parquet file
    # -----------------------------
    for parquet_file in parquet_files:
        print(f"Evaluating {parquet_file} ...")
        local_path = copy_to_local(parquet_file, use_shm=config.data.get("use_shm", False))
        output_data = pd.read_parquet(local_path)

        question_types = output_data[config.data.question_type]
        pred_responses = output_data[config.data.response_key]
        ground_truths = output_data[config.data.ground_truth_key]
        
        # Determine metric column
        if 'metric' in output_data.columns:
            metric_column = 'metric'
        elif 'metrics' in output_data.columns:
            metric_column = 'metrics'
        metrics = output_data[metric_column]

        total = len(output_data)

        # Create Ray tasks
        remote_tasks = [
            process_item.remote(config, question_types[i], metrics[i], pred_responses[i], ground_truths[i])
            for i in range(total)
        ]

        # Collect results
        with tqdm(total=total) as pbar:
            while len(remote_tasks) > 0:
                done_ids, remote_tasks = ray.wait(remote_tasks)
                for result_id in done_ids:
                    question_type, metric, score = ray.get(result_id)
                    all_rewards[question_type][metric].append(score)
                    pbar.update(1)

    # -----------------------------
    # Output results and generate JSON
    # -----------------------------
    print("\n===== Evaluation Results =====")
    results_json = {"results": {}, "overall_mean": None}
    overall_scores = []

    for qtype, mres in all_rewards.items():
        results_json["results"][qtype] = {}

        for metric, values in mres.items():
            avg = float(np.mean(values))
            overall_scores.append(avg)

            # print to console
            print(f"{qtype:30s} | {metric:10s} | mean = {avg:.4f}")

            # save to JSON
            results_json["results"][qtype][metric] = avg
            results_json["results"][qtype][metric + "_all_scores"] = values  # optional raw scores

    # overall mean score
    if len(overall_scores) > 0:
        overall_mean = float(np.mean(overall_scores))
        results_json["overall_mean"] = overall_mean
        print("------------------------------------")
        print(f"Overall Mean Score = {overall_mean:.4f}")

    print("====================================")

    # -----------------------------
    # Write JSON file
    # -----------------------------
    output_file = os.path.join(config.data.path, "evaluation_results.json")

    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=4)

    print(f"\nSaved evaluation results to: {output_file}\n")


if __name__ == "__main__":
    main()