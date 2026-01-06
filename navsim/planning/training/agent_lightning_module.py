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
import pickle
import traceback
from typing import Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
# from navsim.agents.dp.dp_agent import DPAgent
from navsim.agents.gtrs_dense.gtrs_agent import GTRSAgent
from navsim.agents.gtrs_dense.hydra_features import state2traj
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self,
                 agent: AbstractAgent,
                 combined: bool = False,
                 ):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.combined = combined
        self.agent = agent
        self.v_params = get_pacifica_parameters()
        # try:
        #     self.dp_preds_2hz = pickle.load(open(os.getenv('DP_PREDS'), 'rb'))
        # except Exception:
        #     traceback.print_exc()
        #     self.dp_preds_2hz = dict()

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens = batch

        prediction = self.agent.forward(features)

        if isinstance(self.agent, TransfuserAgent):
            loss, loss_dict = self.agent.compute_loss(features, targets, prediction)
        else:
            loss, loss_dict = self.agent.compute_loss(features, targets, prediction, tokens)

        for k, v in loss_dict.items():
            self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

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

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def predict_step(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        if self.combined:
            return self.predict_step_combined(batch, batch_idx)
        if isinstance(self.agent, GTRSAgent):
            return self.predict_step_hydra(batch, batch_idx)
        # elif isinstance(self.agent, DPAgent):
        #     return self.predict_step_dp_traj(batch, batch_idx)
        else:
            raise ValueError('unsupported agent')

    # def predict_step_dp_traj(
    #         self,
    #         batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
    #         batch_idx: int
    # ):
    #     features, targets, tokens = batch
    #     self.agent.eval()
    #     with torch.no_grad():
    #         predictions = self.agent.forward(features)
    #         all_trajs = predictions["dp_pred"]

    #     # interpolate them to 10Hz
    #     all_interpolated_proposals = []
    #     for batch_idx, token in enumerate(tokens):
    #         ego_state = EgoState.build_from_rear_axle(
    #             StateSE2(*features['ego_pose'].cpu().numpy()[batch_idx]),
    #             tire_steering_angle=0.0,
    #             vehicle_parameters=self.v_params,
    #             time_point=TimePoint(0),
    #             rear_axle_velocity_2d=StateVector2D(
    #                 *features['ego_velocity'].cpu().numpy()[batch_idx]
    #             ),
    #             rear_axle_acceleration_2d=StateVector2D(
    #                 *features['ego_acceleration'].cpu().numpy()[batch_idx]
    #             ),
    #         )
    #         interpolated_proposals = []
    #         proposals = all_trajs[batch_idx].cpu().numpy()
    #         for proposal in proposals:
    #             traj = Trajectory(
    #                 proposal,
    #                 TrajectorySampling(time_horizon=4, interval_length=0.5)
    #             )
    #             trans_traj = transform_trajectory(
    #                 traj, ego_state
    #             )
    #             interpolated_traj = get_trajectory_as_array(
    #                 trans_traj,
    #                 TrajectorySampling(num_poses=40, interval_length=0.1),
    #                 ego_state.time_point
    #             )
    #             final_traj = state2traj(interpolated_traj)
    #             interpolated_proposals.append(final_traj)
    #         interpolated_proposals = np.array(interpolated_proposals)
    #         interpolated_proposals = torch.from_numpy(interpolated_proposals).float()[None]
    #         all_interpolated_proposals.append(interpolated_proposals)
    #     # B, 100, 40, 3
    #     all_interpolated_proposals = torch.cat(all_interpolated_proposals, 0)

    #     result = {}
    #     for idx, (proposals, interp_proposals, token) in enumerate(zip(
    #             all_trajs.cpu().numpy(),
    #             all_interpolated_proposals.cpu().numpy(),
    #             tokens
    #     )):
    #         # randomly choose a dp proposal
    #         final_pose = proposals[0]
    #         if final_pose.shape[0] == 40:
    #             interval_length = 0.1
    #         else:
    #             interval_length = 0.5

    #         result[token] = {
    #             'trajectory': Trajectory(final_pose,
    #                                      TrajectorySampling(time_horizon=4, interval_length=interval_length)),
    #             'proposals': proposals,
    #             'interpolated_proposal': interp_proposals
    #         }
    #     return result

    def predict_step_hydra(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"].cpu().numpy()

            imis = predictions["imi"].cpu()
            no_at_fault_collisions_all = predictions["no_at_fault_collisions"].cpu()
            drivable_area_compliance_all = predictions["drivable_area_compliance"].cpu()
            time_to_collision_within_bound_all = predictions["time_to_collision_within_bound"].cpu()
            ego_progress_all = predictions["ego_progress"].cpu()
            driving_direction_compliance_all = predictions["driving_direction_compliance"].cpu()
            lane_keeping_all = predictions["lane_keeping"].cpu()
            traffic_light_compliance_all = predictions["traffic_light_compliance"].cpu()
            score_all = predictions['scores'].cpu()

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

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
             score,
             token) in \
                zip(poses,
                    imis,
                    no_at_fault_collisions_all,
                    drivable_area_compliance_all,
                    time_to_collision_within_bound_all,
                    ego_progress_all,
                    driving_direction_compliance_all,
                    lane_keeping_all,
                    traffic_light_compliance_all,
                    score_all,
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
                'score': score
            }


        return result

    # def predict_step_combined(
    #         self,
    #         batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
    #         not_used: int
    # ):
        features, targets, tokens = batch
        device = features['ego_pose'].device
        all_interpolated_proposals = []
        for batch_idx, token in enumerate(tokens):
            proposals = self.dp_preds_2hz[token]['proposals']
            ego_state = EgoState.build_from_rear_axle(
                StateSE2(*features['ego_pose'].cpu().numpy()[batch_idx]),
                tire_steering_angle=0.0,
                vehicle_parameters=self.v_params,
                time_point=TimePoint(0),
                rear_axle_velocity_2d=StateVector2D(
                    *features['ego_velocity'].cpu().numpy()[batch_idx]
                ),
                rear_axle_acceleration_2d=StateVector2D(
                    *features['ego_acceleration'].cpu().numpy()[batch_idx]
                ),
            )
            interpolated_proposals = []
            for proposal in proposals:
                traj = Trajectory(
                    proposal,
                    TrajectorySampling(time_horizon=4, interval_length=0.5)
                )
                trans_traj = transform_trajectory(
                    traj, ego_state
                )
                interpolated_traj = get_trajectory_as_array(
                    trans_traj,
                    TrajectorySampling(num_poses=40, interval_length=0.1),
                    ego_state.time_point
                )
                final_traj = state2traj(interpolated_traj)
                interpolated_proposals.append(final_traj)
            interpolated_proposals = np.array(interpolated_proposals)
            interpolated_proposals = torch.from_numpy(interpolated_proposals).float().to(device)[None]
            all_interpolated_proposals.append(interpolated_proposals)
        # B, 100, 40, 3
        all_interpolated_proposals = torch.cat(all_interpolated_proposals, 0)

        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.evaluate_dp_proposals(features, all_interpolated_proposals)
            poses = predictions["trajectory"].cpu().numpy()
            imis = predictions["imi"].cpu()
            no_at_fault_collisions_all = predictions["no_at_fault_collisions"].cpu()
            drivable_area_compliance_all = predictions["drivable_area_compliance"].cpu()
            time_to_collision_within_bound_all = predictions["time_to_collision_within_bound"].cpu()
            ego_progress_all = predictions["ego_progress"].cpu()
            driving_direction_compliance_all = predictions["driving_direction_compliance"].cpu()
            lane_keeping_all = predictions["lane_keeping"].cpu()
            traffic_light_compliance_all = predictions["traffic_light_compliance"].cpu()
            score_all = predictions['scores'].cpu()

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        result = {}
        for (pose,
             interpolated_proposal,
             imi,
             no_at_fault_collisions,
             drivable_area_compliance,
             time_to_collision_within_bound,
             ego_progress,
             driving_direction_compliance,
             lane_keeping,
             traffic_light_compliance,
             score,
             token) in zip(poses,
                           all_interpolated_proposals.cpu().numpy(),
                           imis,
                           no_at_fault_collisions_all,
                           drivable_area_compliance_all,
                           time_to_collision_within_bound_all,
                           ego_progress_all,
                           driving_direction_compliance_all,
                           lane_keeping_all,
                           traffic_light_compliance_all,
                           score_all,
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
                'interpolated_proposal': interpolated_proposal,
                'score': score,
            }
        return result