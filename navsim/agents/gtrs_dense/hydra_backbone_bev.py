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

import torch
from torch import nn

from navsim.agents.backbones.vov import VoVNet
from navsim.agents.gtrs_dense.hydra_config import HydraConfig


class HydraBackboneBEV(nn.Module):

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
            raise ValueError('Unsupported vision backbone')

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        self.img_feat_c = vit_channels

        self.bev_h, self.bev_w = 8, 8
        img_num = 2 if self.config.use_back_view else 1
        self.bev_queries = nn.Embedding(
            self.bev_h * self.bev_w, self.img_feat_c
        )
        self.pos_emb = nn.Embedding(
            config.img_vert_anchors * config.img_horz_anchors * img_num + self.bev_h * self.bev_w, self.img_feat_c
        )

        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.img_feat_c,
                nhead=16,
                dim_feedforward=self.img_feat_c * 4,
                dropout=0.0,
                batch_first=True
            ), self.config.fusion_layers
        )

        channel = self.config.bev_features_channels
        self.relu = nn.ReLU(inplace=True)
        # top down
        if self.config.detect_boxes or self.config.use_bev_semantic:
            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_resolution_height
                    // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width
                    // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

            self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

            # lateral
            self.c5_conv = nn.Conv2d(
                self.img_feat_c,
                channel,
                (1, 1),
            )

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))
        return p3

    def encode_img(self, img):
        B, C, H, W = img.shape
        if self.backbone_type == 'vov':
            image_features = self.image_encoder(img)[-1]
        else:
            raise ValueError('Forward wrong backbone')
        img_tokens = self.avgpool_img(image_features)
        return img_tokens.flatten(-2, -1).permute(0, 2, 1)

    def forward(self, image_front, image_back):
        B = image_front.shape[0]

        image_features_front = self.encode_img(image_front)
        image_features_back = self.encode_img(image_back)

        img_tokens = torch.cat([image_features_front, image_features_back], 1)
        bev_tokens = self.bev_queries.weight[None].repeat(B, 1, 1)
        img_len = img_tokens.shape[1]
        tokens = torch.cat([
            img_tokens, bev_tokens
        ], 1)

        tokens = self.fusion_encoder(
            tokens + self.pos_emb.weight[None].repeat(B, 1, 1)
        )
        img_tokens_ = tokens[:, :img_len]
        bev_tokens_ = tokens[:, img_len:]

        up_bev_tokens = self.top_down(
            bev_tokens_.permute(0, 2, 1).view(B, self.img_feat_c, self.bev_h, self.bev_w)
        )
        return img_tokens_, bev_tokens_, up_bev_tokens
