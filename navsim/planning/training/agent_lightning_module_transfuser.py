import pytorch_lightning as pl

import torch
from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens= batch
        prediction = self.agent.forward(features)
        loss = self.agent.compute_loss(features, targets, prediction)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        
        features, targets, tokens = batch

        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"].cpu().numpy()
        
        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5
        print(f"Interval length: {interval_length}")

        result = {}
        for (pose, token) in zip(poses, tokens):
            result[token] = {"trajectory": Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length))}

        return result

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()