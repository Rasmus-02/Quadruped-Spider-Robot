# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def corrupted_height_scan(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    noise_std: float = 0.03, 
    dropout_prob: float = 0.1
) -> torch.Tensor:
    """
    Samples the height scan and applies synthetic noise and dropouts to simulate 
    real-world elevation mapping inaccuracies.
    """
    sensor: RayCaster = env.scene[sensor_cfg.name]
    
    # Compute relative Z-heights from the sensor to the ground hits
    hit_pos_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w
    
    # Isaac Lab returns large values or zeros for rays that miss. 
    # We identify valid hits to avoid exploding math.
    valid_hits = (hit_pos_w[..., 2] != 0.0)
    
    heights = torch.zeros_like(hit_pos_w[..., 2])
    heights[valid_hits] = hit_pos_w[..., 2][valid_hits] - sensor_pos_w[..., 2].unsqueeze(1).expand_as(heights)[valid_hits]
    
    # Add Gaussian noise (simulates depth camera jitter and odometry drift)
    noise = torch.randn_like(heights) * noise_std
    heights += noise
    
    # Add dropouts (simulates occlusions, shadows, and sensor blindness)
    dropout_mask = torch.rand_like(heights) < dropout_prob
    heights[dropout_mask] = -1.0  # Assign a fixed penalty value to blind spots
    
    # Clip to a normalized range for the neural network
    return torch.clamp(heights, min=-1.0, max=1.0)