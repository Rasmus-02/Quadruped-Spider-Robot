# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import height_scan
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def corrupted_height_scan(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    offset: float = 0.5,
    noise_std: float = 0.015, 
    dropout_prob: float = 0.05,
    blackout_prob: float = 0.2,
) -> torch.Tensor:
    """
    Samples the height scan matching Isaac Lab's mdp.height_scan convention
    and applies synthetic noise, pixel dropouts, and full-sensor blackouts.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    hit_pos_w = sensor.data.ray_hits_w[..., 2]
    sensor_pos_w = sensor.data.pos_w[:, 2].unsqueeze(1)
    
    heights = sensor_pos_w - hit_pos_w - offset
    heights = torch.nan_to_num(heights, nan=1.0, posinf=1.0, neginf=-1.0)
    
    if noise_std > 0.0:
        heights = heights + torch.randn_like(heights) * noise_std
    
    # Salt-and-pepper pixel dropouts
    if dropout_prob > 0.0:
        dropout_mask = torch.rand_like(heights) < dropout_prob
        heights[dropout_mask] = 1.0

    # Full sensor blackouts
    if blackout_prob > 0.0:
        # 1D mask per environment (True = blinded)
        blackout_mask = torch.rand(heights.shape[0], 1, device=heights.device) < blackout_prob
        # Overwrite the entire 121-ray scan with 1.0 for blinded environments
        heights = torch.where(blackout_mask, torch.ones_like(heights), heights)
    
    return torch.clamp(heights, min=-1.0, max=1.0)