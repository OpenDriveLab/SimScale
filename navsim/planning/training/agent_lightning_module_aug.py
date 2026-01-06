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

import os
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.gtrs_aug.gtrs_aug_agent import GTRSAugAgent
from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug
from navsim.agents.gtrs_aug.utils.util import CosineScheduler
from navsim.common.dataclasses import Trajectory


class AgentLightningModuleAug(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self,
                 cfg: HydraConfigAug,
                 agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()

        self._cfg = cfg
        self.agent: GTRSAugAgent = agent
        self.only_ori_input = cfg.only_ori_input
        self.n_rotation_crop = cfg.student_rotation_ensemble

        if self._cfg.lab.use_cosine_ema_scheduler:
            self.momentum_schedule = CosineScheduler(self._cfg.lab.ema_momentum_start, 0.999, 10)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens = batch
        teacher_pred, student_preds, loss_dict = self.agent.forward(batch)
        loss_student = self.agent.compute_loss(features, targets, student_preds, tokens)

        ori_loss = loss_student['ori']
        for k, v in ori_loss[1].items():
            self.log(f"{logging_prefix}/{k}-ori", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/loss-ori", ori_loss[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        loss = ori_loss[0]

        if not self.only_ori_input:
            aug_loss = loss_student['aug']
            for k, v in aug_loss[1].items():
                self.log(f"{logging_prefix}/{k}-aug", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss-aug", aug_loss[0], on_step=True, on_epoch=True, prog_bar=True,
                     sync_dist=True)
            loss = loss + aug_loss[0]

        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()

        if not self._cfg.lab.ban_soft_label_loss:
            loss_soft_teacher = self.agent.compute_loss_soft_teacher(teacher_pred, student_preds[0], targets, tokens)
            for k, v in loss_soft_teacher[1].items():
                self.log(f"{logging_prefix}/{k}-soft", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss-soft", loss_soft_teacher[0], on_step=True, on_epoch=True, prog_bar=True,
                     sync_dist=True)
            loss = loss + loss_soft_teacher[0]

        if self._cfg.refinement.use_multi_stage:
            loss_refinement = self.agent.compute_loss_multi_stage(features, targets, student_preds, tokens)

            loss_refinement_ori = loss_refinement['ori']
            for k, v in loss_refinement_ori[1].items():
                self.log(f"{logging_prefix}/{k}-ori", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss-refinement_ori", loss_refinement_ori[0], on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            loss = loss + loss_refinement_ori[0]

            if not self.only_ori_input:
                loss_refinement_aug = loss_refinement['aug']
                for k, v in loss_refinement_aug[1].items():
                    self.log(f"{logging_prefix}/{k}-aug", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/loss-refinement_aug", loss_refinement_aug[0], on_step=True, on_epoch=True,
                         prog_bar=True, sync_dist=True)
                loss = loss + loss_refinement_aug[0]

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

    def on_train_start(self):
        self.agent.model.train()

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer,
            optimizer_closure=None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb;
            pdb.set_trace()
        # if self._cfg.lab.use_cosine_ema_scheduler:
        #     m = self.momentum_schedule[epoch]

        if self._cfg.backbone_type in ('resnet34', 'resnet50'):
            if epoch < 3:
                m = 0
            elif epoch < 6:
                m = 0.992 + (epoch - 3) * 0.002
            else:
                m = 0.998
        else:
            if epoch < 3:
                m = 0.992 + epoch * 0.002
            else:
                m = 0.998
        self.agent.model.update_teacher(m)

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def predict_step(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        return self.predict_step_hydra(batch, batch_idx)

    def predict_step_hydra(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions, _, _ = self.agent.forward(batch)
            if self._cfg.refinement.use_multi_stage and not self._cfg.lab.use_first_stage_traj_in_infer:
                poses = predictions['final_traj'].cpu().numpy()
            else:
                poses = predictions["trajectory"].cpu().numpy()
            imis = predictions["imi"].softmax(-1).cpu()
            no_at_fault_collisions_all = predictions["no_at_fault_collisions"].cpu()
            drivable_area_compliance_all = predictions["drivable_area_compliance"].cpu()
            time_to_collision_within_bound_all = predictions[
                "time_to_collision_within_bound"].cpu()
            ego_progress_all = predictions["ego_progress"].cpu()
            driving_direction_compliance_all = predictions["driving_direction_compliance"].cpu()
            lane_keeping_all = predictions["lane_keeping"].cpu()
            traffic_light_compliance_all = predictions["traffic_light_compliance"].cpu()
            # history_comfort_all = predictions["history_comfort"].sigmoid().log().cpu().numpy()
            # score_all = predictions['scores'].cpu()

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb;
            pdb.set_trace()

        filtered_trajs_all = predictions['refinement'][0]['trajs'].cpu().numpy()
        filtered_scores = predictions['refinement'][0]['scores'].cpu().numpy()
        result = {}
        for (pose,
             imi,
             no_at_fault_collisions,
             drivable_area_compliance,
             time_to_collision_within_bound,
             ego_progress,
             driving_direction_compliance,
             lane_keeping,
             traffic_light_compliance,
             filtered_traj,
             filtered_score,
            #  score,
             token) in zip(poses,
                           imis,
                           no_at_fault_collisions_all,
                           drivable_area_compliance_all,
                           time_to_collision_within_bound_all,
                           ego_progress_all,
                           driving_direction_compliance_all,
                           lane_keeping_all,
                           traffic_light_compliance_all,
                           filtered_trajs_all,
                           filtered_scores,
                        #    score_all,
                           tokens):
            result[token] = {
                'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                'imi': imi,
                'no_at_fault_collisions': no_at_fault_collisions,
                'drivable_area_compliance': drivable_area_compliance,
                'time_to_collision_within_bound': time_to_collision_within_bound,
                'ego_progress': ego_progress,
                'driving_direction_compliance': driving_direction_compliance,
                'lane_keeping': lane_keeping,
                'traffic_light_compliance': traffic_light_compliance,
                'filtered_traj': filtered_traj,
                'filtered_score': filtered_score,
                # 'score': score,
            }
        return result
