# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import math

import numpy as np
import torch


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)

    return pos_x


def nerf_positional_encoding(
        tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def traj2nerf(traj):
    result = torch.cat(
        [
            nerf_positional_encoding(traj[..., :2]),
            torch.cos(traj[..., -1])[..., None],
            torch.sin(traj[..., -1])[..., None],
        ], dim=-1
    )
    return result


def nerf2traj(nerf, num_encoding_functions=6, include_input=False, log_sampling=True):
    # Calculate the length of the original 2D position tensor
    original_dim = 2

    # Calculate the length of the positional encoding for the 2D position tensor
    if include_input:
        encoding_length = original_dim * (2 * num_encoding_functions + 1)
    else:
        encoding_length = original_dim * 2 * num_encoding_functions

    # Extract the positional encoding for the original 2D position tensor
    positional_encoding = nerf[..., :encoding_length]

    # Reverse positional encoding
    if include_input:
        original_position = positional_encoding[..., :original_dim]
        positional_encoding = positional_encoding[..., original_dim:]
    else:
        original_position = torch.zeros(
            (*nerf.shape[:-1], original_dim), dtype=nerf.dtype, device=nerf.device
        )

    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=nerf.dtype,
            device=nerf.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=nerf.dtype,
            device=nerf.device,
        )

    for i, freq in enumerate(frequency_bands):
        for j, func in enumerate([torch.sin, torch.cos]):
            original_position += func(positional_encoding[..., (2 * i + j)::2 * num_encoding_functions]) / freq

    # Extract the sine and cosine of the angle
    cos_angle = nerf[..., -2]
    sin_angle = nerf[..., -1]

    # Reconstruct the angle using atan2
    angle = torch.atan2(sin_angle, cos_angle)

    # Combine the original position and the angle to form the trajectory
    traj = torch.cat([original_position, angle[..., None]], dim=-1)
    return traj


if __name__ == '__main__':
    traj = torch.from_numpy(np.load('/mnt/f/e2e/navsim_ours/traj_final/test_4096_kmeans.npy'))
    nerf = traj2nerf(traj)
    traj_2 = nerf2traj(nerf)
