# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pickle
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.run_pdm_score import create_scene_aggregators, calculate_individual_mapping_scores, \
    compute_final_scores
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    model_trajectory = {k:v for a in args for k,v in a["model_trajectory"].items()}

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
            simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
    )

    pdm_results: List[pd.DataFrame] = []

    # first stage

    traffic_agents_policy_stage_one: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )

    scene_loader_tokens_stage_one = scene_loader.tokens_stage_one

    tokens_to_evaluate_stage_one = list(set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens))
    for idx, (token) in enumerate(tokens_to_evaluate_stage_one):
        logger.info(
            f"Processing stage one reactive scenario {idx + 1} / {len(tokens_to_evaluate_stage_one)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            # trajectory = metric_cache.human_trajectory
            trajectory = model_trajectory[token]['trajectory']
            score_row_stage_one, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy_stage_one,
            )
            score_row_stage_one["valid"] = True
            score_row_stage_one["log_name"] = metric_cache.log_name
            score_row_stage_one["frame_type"] = metric_cache.scene_type
            score_row_stage_one["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row_stage_one["endpoint_x"] = absolute_endpoint.x
            score_row_stage_one["endpoint_y"] = absolute_endpoint.y
            score_row_stage_one["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row_stage_one["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row_stage_one["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row_stage_one = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_one["valid"] = False
        score_row_stage_one["token"] = token

        pdm_results.append(score_row_stage_one)

    # second stage
    # Note: Disable second stage processing when evaluating on navtrain or navtest

    traffic_agents_policy_stage_two: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )
    scene_loader_tokens_stage_two = scene_loader.reactive_tokens_stage_two

    tokens_to_evaluate_stage_two = list(set(scene_loader_tokens_stage_two) & set(metric_cache_loader.tokens))
    for idx, (token) in enumerate(tokens_to_evaluate_stage_two):
        logger.info(
            f"Processing stage two reactive scenario {idx + 1} / {len(tokens_to_evaluate_stage_two)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = model_trajectory[token]['trajectory']

            score_row_stage_two, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy_stage_two,
            )
            score_row_stage_two["valid"] = True
            score_row_stage_two["log_name"] = metric_cache.log_name
            score_row_stage_two["frame_type"] = metric_cache.scene_type
            score_row_stage_two["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row_stage_two["endpoint_x"] = absolute_endpoint.x
            score_row_stage_two["endpoint_y"] = absolute_endpoint.y
            score_row_stage_two["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row_stage_two["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row_stage_two["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row_stage_two = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_two["valid"] = False
        score_row_stage_two["token"] = token

        pdm_results.append(score_row_stage_two)

    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    combined = cfg.get('combined_inference', False)

    print(f'Combined inference: {combined}')
    dump_path = os.getenv('SUBSCORE_PATH')
    
    # gpu inference
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    scene_loader_inference = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
        is_training=False
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks(),logger=False)
    predictions = trainer.predict(
        AgentLightningModule(
            agent=agent,
            combined=combined
        ),
        dataloader,
        return_predictions=True
    )

    dist.barrier()
    all_predictions = [None for _ in range(dist.get_world_size())]

    if dist.is_initialized():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions.append(predictions)

    if dist.get_rank() != 0:
        return None

    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)

    pickle.dump(merged_predictions, open(dump_path, 'wb'))
    print(f'Subscore/Trajectories saved to {dump_path}')

    merged_predictions = {k:{'trajectory': v['trajectory']} for k, v in merged_predictions.items()}

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": merged_predictions
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    
    del merged_predictions
    
    worker = build_worker(cfg)
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.concat(score_rows)
    pickle.dump(pdm_score_df, open(dump_path.replace('.pkl', '_epdms.pkl'), 'wb'))

    try:
        raw_mapping = cfg.train_test_split.reactive_all_mapping
        all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

        for orig_token, prev_token, two_stage_pairs in raw_mapping:
            if prev_token in set(scene_loader.tokens) or orig_token in set(scene_loader.tokens):
                all_mappings[(orig_token, prev_token)] = [tuple(pair) for pair in two_stage_pairs]

        pdm_score_df = create_scene_aggregators(
            all_mappings, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
        )
        pdm_score_df = compute_final_scores(pdm_score_df)
        pseudo_closed_loop_valid = True

    except Exception:
        logger.warning("----------- Failed to calculate pseudo closed-loop weights or comfort:")
        traceback.print_exc()
        pdm_score_df["weight"] = 1.0
        pseudo_closed_loop_valid = False

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
                (any(score.name in c for score in
                     fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
                and c != "pdm_score"
        )
    ]

    pcl_group_score, pcl_stage1_score, pcl_stage2_score = calculate_individual_mapping_scores(
        pdm_score_df[score_cols + ["token", "weight"]], all_mappings
    )

    for col in score_cols:
        stage_one_mask = pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
        stage_two_mask = pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC

        pdm_score_df.loc[stage_one_mask, f"{col}_stage_one"] = pdm_score_df.loc[stage_one_mask, col]
        pdm_score_df.loc[stage_two_mask, f"{col}_stage_two"] = pdm_score_df.loc[stage_two_mask, col]

    pdm_score_df.drop(columns=score_cols, inplace=True)
    pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(pdm_score_df["score_stage_two"])
    pdm_score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)

    stage1_cols = [f"{col}_stage_one" for col in score_cols if col != "score"]
    stage2_cols = [f"{col}_stage_two" for col in score_cols if col != "score"]
    score_cols = stage1_cols + stage2_cols + ["score"]

    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]

    summary_rows = []

    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = pseudo_closed_loop_valid
    stage1_row["score"] = pcl_stage1_score.get("score", np.nan)
    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            stage1_row[f"{col}_stage_one"] = pcl_stage1_score[col]
    summary_rows.append(stage1_row)

    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = pseudo_closed_loop_valid
    stage2_row["score"] = pcl_stage2_score.get("score", np.nan)
    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            stage2_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(stage2_row)

    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = pseudo_closed_loop_valid
    combined_row["score"] = pcl_group_score["score"]

    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_one"] = pcl_stage1_score[col]

    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(combined_row)

    pdm_score_df = pd.concat([pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True)

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final extended pdm score of valid results: {pdm_score_df[pdm_score_df["token"] == "extended_pdm_score_combined"]["score"].iloc[0]}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )

    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )


if __name__ == "__main__":
    main()
