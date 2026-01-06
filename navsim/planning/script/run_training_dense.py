import datetime
import logging
from pathlib import Path
from typing import Tuple
import os
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name
            for log_name in train_scene_filter.log_names
            if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)
    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    cfg.agent.max_epochs = cfg.trainer.params.max_epochs
    agent: AbstractAgent = instantiate(cfg.agent)

    # NOTE 
    if os.getenv('SYN_IDX', None) is not None:
        agent.load_pdm_socre_syn()

    if agent._checkpoint_path is not None:
        agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
                cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
            split="train"
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            split="val"
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    tf_logger = pl.loggers.TensorBoardLogger(
        save_dir=f"{os.environ.get('NAVSIM_EXP_ROOT')}/tensorboard_logs/{agent.__module__.split('.')[-2]}/{cfg.experiment_name}/",
        name=cfg.experiment_name, 
        version=0, 
        default_hp_metric=False,
    )

    logger.info("Building Trainer")
    if isinstance(agent, TransfuserAgent):
        trainer = pl.Trainer(**cfg.trainer.params,
                             logger=tf_logger,
                             callbacks=agent.get_training_callbacks())
    else:
        if cfg.trainer.params.get('strategy') is not None:
            cfg.trainer.params.__delattr__('strategy')
        trainer = pl.Trainer(**cfg.trainer.params,
                             callbacks=agent.get_training_callbacks(),
                             logger=tf_logger,
                             strategy=DDPStrategy(static_graph=False,
                                        find_unused_parameters=True,
                                        timeout=datetime.timedelta(seconds=3600)))
    
    ckpt_path = None
    if cfg.get('resume_ckpt_path', None):
        resume_ckpt_path = f"{os.environ.get('NAVSIM_EXP_ROOT')}/{cfg.resume_ckpt_path}"
        if os.path.exists(resume_ckpt_path):
            logger.info(f"Resuming from checkpoint {resume_ckpt_path}")
            ckpt_path = resume_ckpt_path

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path
    )


if __name__ == "__main__":
    main()
