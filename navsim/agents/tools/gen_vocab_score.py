import logging
import lzma
import os
import pickle
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple

import hydra
import numpy as np
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.evaluate.pdm_score import pdm_score_full_v2
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

vocab_size = 16384
logger = logging.getLogger(__name__)
trajpdm_root = os.getenv('NAVSIM_TRAJPDM_ROOT')
devkit_root = os.getenv('NAVSIM_DEVKIT_ROOT')
traj_path = f"{devkit_root}/traj_final/{vocab_size}.npy"
dir = f'vocab_score_full_{vocab_size}_{os.getenv("split", "navtrain")}_{os.getenv("POSTFIX")}'
CONFIG_PATH = f"{devkit_root}/navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)
    vocab = np.load(traj_path)
    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_filter = instantiate(cfg.train_test_split.scene_filter),
    if isinstance(scene_filter, tuple) and len(scene_filter) == 1:
        scene_filter = scene_filter[0]
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
    )
    os.makedirs(f'{trajpdm_root}/{dir}', exist_ok=True)
    result_path = f'{trajpdm_root}/{dir}/{cfg.save_name}.pkl'
    print(f'Results will be written to {result_path}')

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "vocab": vocab
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    new_data_points = []
    for data in data_points:
        for token in data['tokens']:
            new_data_points.append({
                "cfg": cfg,
                "log_file": data['log_file'],
                "token": token,
                "vocab": vocab
            })

    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, new_data_points)
    final = {}
    for tmp in score_rows:
        final[tmp['token']] = tmp['score']
    pickle.dump(final, open(result_path, 'wb'))


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    # tokens = [t for a in args for t in a["tokens"]]
    tokens = [a["token"] for a in args]
    cfg: DictConfig = args[0]["cfg"]
    vocab_trajectories = args[0]["vocab"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
    )
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token}
        try:
            tmp_cache_path = f'{trajpdm_root}/{dir}/{token}/tmp.pkl'
            if os.path.exists(tmp_cache_path):
                print(f'Exists: {tmp_cache_path}')
                # load cache
                score_row['score'] = pickle.load(open(tmp_cache_path, 'rb'))
                pdm_results.append(score_row)
                continue

            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            # transform vocab into traj
            pdm_result = pdm_score_full_v2(
                metric_cache=metric_cache,
                vocab_trajectories=vocab_trajectories,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )

            score_row['score'] = pdm_result
            #     save cache
            os.makedirs(tmp_cache_path.replace('tmp.pkl', ''), exist_ok=True)
            pickle.dump(pdm_result, open(tmp_cache_path, 'wb'))

        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            return None

        pdm_results.append(score_row)
    return pdm_results


if __name__ == "__main__":
    main()