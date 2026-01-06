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
from dataclasses import dataclass
from typing import Tuple

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_config import TransfuserConfig

NAVSIM_DEVKIT_ROOT = os.environ.get("NAVSIM_DEVKIT_ROOT")


@dataclass
class HydraConfig(TransfuserConfig):
    scheduler: str = 'warmup_cos'  # 'warmup_cos, default

    version: str = 'default'
    use_temporal_bev_kv: bool = False

    regression_ep: bool = False
    three2two: bool = True

    vocab_dropout: bool = False

    use_hist_ego_status: bool = False

    seq_len: int = 2
    trajectory_imi_weight: float = 1.0

    if regression_ep:
        ep_lw = 50.0
    else:
        ep_lw = 2.0
    
    trajectory_pdm_weight = {
            'no_at_fault_collisions': 3.0,
            'drivable_area_compliance': 3.0,
            'time_to_collision_within_bound': 4.0,
            'ego_progress': ep_lw,
            'driving_direction_compliance': 1.0,
            'lane_keeping': 2.0,
            'traffic_light_compliance': 3.0,
            'history_comfort': 1.0,
        }

    progress_weight: float = 2.0
    ttc_weight: float = 2.0

    syn_imi: bool = True

    inference_imi_weight: float = 0.1
    inference_da_weight: float = 1.0
    decouple: bool = False
    vocab_size: int = 4096
    vocab_path: str = None
    normalize_vocab_pos: bool = False
    num_ego_status: int = 1
    fusion_layers: int = 3

    ckpt_path: str = None
    sigma: float = 0.5
    use_pers_bev_embed: bool = False
    type: str = 'center'
    rel: bool = False
    use_nerf: bool = False
    extra_traj_layer: bool = False

    use_back_view: bool = False
    use_lr_view: bool = False

    extra_tr: bool = False
    vadv2_head_nhead: int = 8
    vadv2_head_nlayers: int = 3

    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=4, interval_length=0.1
    )

    # img backbone
    use_final_fpn: bool = False
    use_img_pretrained: bool = False
    image_architecture: str = "resnet34"
    bkb_path: str = ''
    
    # lidar backbone
    lidar_architecture: str = "resnet34"
    latent: bool = True

    backbone_type: str = 'resnet'
    vit_ckpt: str = ''
    intern_ckpt: str = ''
    vov_ckpt: str = ''
    eva_ckpt: str = ''
    swin_ckpt: str = ''

    sptr_ckpt: str = ''
    map_ckpt: str = ''

    lr_mult_backbone: float = 1.0
    backbone_wd: float = 0.0
    weight_decay: float = 0.0
    
    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 2048
    camera_height: int = 512
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = camera_height // 32
    img_horz_anchors: int = camera_width // 32
    lidar_vert_anchors: int = lidar_resolution_height // 32
    lidar_horz_anchors: int = lidar_resolution_width // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic: bool = False # NOTE default to False
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 1 / pixels_per_meter

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
