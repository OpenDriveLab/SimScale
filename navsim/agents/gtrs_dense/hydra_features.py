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

from enum import IntEnum
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import affinity
from shapely.geometry import Polygon, LineString
from torchvision import transforms

from navsim.agents.gtrs_dense.hydra_config import HydraConfig
from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


def state2traj(states):
    rel_poses = absolute_to_relative_poses([StateSE2(*tmp) for tmp in
                                            states[:, StateIndex.STATE_SE2]])
    final_traj = [pose.serialize() for pose in rel_poses[1:]]
    return final_traj


class HydraFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: HydraConfig):
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["camera_feature_back"] = self._get_camera_feature_back_multiple(agent_input)
        if self._config.use_lr_view:
            features["camera_feature_left"] = self._get_camera_feature_single(agent_input, 'left')
            features["camera_feature_right"] = self._get_camera_feature_single(agent_input, 'right')

        ego_status_list = []
        for i in range(self._config.num_ego_status): # default is 1 NOTE GTRS only use this
            idx = -(i + 1)
            ego_status = torch.concatenate(
                [
                    torch.tensor(agent_input.ego_statuses[idx].driving_command, dtype=torch.float32),# 3
                    torch.tensor(agent_input.ego_statuses[idx].ego_velocity, dtype=torch.float32), # 2
                    torch.tensor(agent_input.ego_statuses[idx].ego_acceleration, dtype=torch.float32) # 2
                ],
            )

            ego_status_list.append(ego_status)

        if self._config.use_hist_ego_status: # default is False
            hist_es_list = []
            for es in agent_input.ego_statuses[:-1]:
                ego_status_hist = torch.concatenate(
                    [
                        torch.tensor(es.ego_velocity, dtype=torch.float32),
                        torch.tensor(es.ego_acceleration, dtype=torch.float32),
                        torch.tensor(es.ego_pose, dtype=torch.float32),
                    ],
                )
                hist_es_list.append(ego_status_hist)
            features["hist_status_feature"] = torch.cat(hist_es_list)
        features["status_feature"] = ego_status_list
        features['ego_pose'] = agent_input.ego_statuses[-1].ego_pose
        features['ego_velocity'] = agent_input.ego_statuses[-1].ego_velocity
        features['ego_acceleration'] = agent_input.ego_statuses[-1].ego_acceleration
        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        seq_len = self._config.seq_len # default is 2
        cameras = agent_input.cameras[-seq_len:] 
        assert (len(cameras) == seq_len)
        image_list = []
        for camera in cameras:
            image = camera.cam_l0.image
            if image is not None and image.size > 0 and np.any(image):
                l0 = camera.cam_l0.image[28:-28, 416:-416]
                f0 = camera.cam_f0.image[28:-28]
                r0 = camera.cam_r0.image[28:-28, 416:-416]
                # stitch l0, f0, r0 images
                stitched_image = np.concatenate([l0, f0, r0], axis=1)
                resized_image = cv2.resize(stitched_image, (self._config.camera_width, self._config.camera_height))
                tensor_image = transforms.ToTensor()(resized_image)
                image_list.append(tensor_image)
            else:
                image_list.append(None)

        return image_list

    def _get_camera_feature_single(self, agent_input: AgentInput, view) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        seq_len = self._config.seq_len
        cameras = agent_input.cameras[-seq_len:]
        image_list = []
        for camera in cameras:
            image = camera.cam_l0.image
            if image is not None and image.size > 0 and np.any(image):
                if view == 'left':
                    single_img = camera.cam_l1.image[28:-28]
                elif view == 'right':
                    single_img = camera.cam_r1.image[28:-28]
                else:
                    raise ValueError('Unsupported view')
                resized_image = cv2.resize(single_img,
                                           (int(self._config.camera_width * 1920 / 4096), self._config.camera_height))
                tensor_image = transforms.ToTensor()(resized_image)
                image_list.append(tensor_image)
            else:
                image_list.append(None)

        return image_list

    def _get_camera_feature_back_multiple(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        seq_len = self._config.seq_len
        cameras = agent_input.cameras[-seq_len:]
        assert (len(cameras) == seq_len)
        image_list = []
        for camera in cameras:
            image = camera.cam_l0.image
            if image is not None and image.size > 0 and np.any(image):
                l2 = camera.cam_l2.image[28:-28, 416:-416]
                b0 = camera.cam_b0.image[28:-28]
                r2 = camera.cam_r2.image[28:-28, 416:-416]
                # stitch l0, f0, r0 images
                stitched_image = np.concatenate([l2, b0, r2], axis=1)
                resized_image = cv2.resize(stitched_image, (self._config.camera_width, self._config.camera_height))
                tensor_image = transforms.ToTensor()(resized_image)
                image_list.append(tensor_image)
            else:
                image_list.append(None)

        return image_list


class HydraTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config: HydraConfig):
        self._config = config
        self.v_params = get_pacifica_parameters()
        self.simulator = PDMSimulator(
            TrajectorySampling(num_poses=40, interval_length=0.1)
        )

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_traj = scene.get_future_trajectory(
            num_trajectory_frames=int(4 // 0.5)
        )
        trajectory = torch.tensor(future_traj.poses)
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

        ego_state = EgoState.build_from_rear_axle(
            StateSE2(*scene.frames[frame_idx].ego_status.ego_pose),
            tire_steering_angle=0.0,
            vehicle_parameters=self.v_params,
            time_point=TimePoint(scene.frames[frame_idx].timestamp),
            rear_axle_velocity_2d=StateVector2D(
                *scene.frames[frame_idx].ego_status.ego_velocity
            ),
            rear_axle_acceleration_2d=StateVector2D(
                *scene.frames[frame_idx].ego_status.ego_acceleration
            ),
        )
        trans_traj = transform_trajectory(
            future_traj, ego_state
        )
        interpolated_traj = get_trajectory_as_array(
            trans_traj,
            TrajectorySampling(num_poses=40, interval_length=0.1),
            ego_state.time_point
        )
        final_traj = state2traj(interpolated_traj)
        final_traj = torch.tensor(final_traj)
        return {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
            "interpolated_traj": final_traj,
        }

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: HydraConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                    config.lidar_min_y <= y <= config.lidar_max_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
            self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _geometry_local_coords(self, geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)

    def _compute_map_polygon_mask(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
            self, annotations: Annotations, layers: TrackedObjectType
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw)
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self, layers=layers
        )
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects


class BoundingBox2DIndex(IntEnum):
    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
               and not attribute.startswith("__")
               and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
