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

from torch import nn

from navsim.agents.backbones.vov import VoVNet
from navsim.agents.gtrs_dense.hydra_config import HydraConfig


class HydraBackbone(nn.Module):
    def __init__(self, config: HydraConfig):

        super().__init__()
        self.config = config
        self.backbone_type = config.backbone_type
        if config.backbone_type == 'vov':
            self.image_encoder = VoVNet(
                spec_name='V-99-eSE',
                out_features=['stage4', 'stage5'],
                norm_eval=True,
                with_cp=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint=config.vov_ckpt,
                    prefix='img_backbone.'
                )
            )
            vit_channels = 1024
            self.image_encoder.init_weights()
        else:
            raise ValueError('Unsupported backbone')

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        self.img_feat_c = vit_channels

    def forward(self, image):
        B, C, H, W = image.shape
        if self.backbone_type == 'vov':
            image_features = self.image_encoder(image)[-1]
        else:
            raise ValueError('Forward wrong backbone')
        return self.avgpool_img(image_features)
