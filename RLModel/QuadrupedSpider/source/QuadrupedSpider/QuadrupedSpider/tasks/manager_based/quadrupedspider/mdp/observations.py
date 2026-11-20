# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_yaw

from .events import ensure_obstacle_bounds_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FOOT_BODY_NAMES = ("FLTibia_1", "FRTibia_1", "RLTibia_1", "RRTibia_1")
FOOT_SCAN_RADII = (0.08, 0.16, 0.26, 0.36, 0.48)
FOOT_SCAN_COUNTS = (6, 8, 10, 12, 16)
NUM_FOOT_SAMPLES = sum(FOOT_SCAN_COUNTS)  # 52 per foot, 208 total


def _height_grid_shape(sensor: RayCaster) -> tuple[int, int, float, float, float]:
    size_x, size_y = sensor.cfg.pattern_cfg.size
    resolution = sensor.cfg.pattern_cfg.resolution
    num_x = round(size_x / resolution) + 1
    num_y = round(size_y / resolution) + 1
    if sensor.num_rays != num_x * num_y:
        raise ValueError(
            f"Expected a regular {num_x}x{num_y} height grid ({num_x * num_y} rays), "
            f"got {sensor.num_rays} rays."
        )
    return num_x, num_y, size_x, size_y, resolution


def _foot_scan_offsets(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create the 52-point concentric ring pattern around one foot."""
    rings = []
    for radius, count in zip(FOOT_SCAN_RADII, FOOT_SCAN_COUNTS):
        angles = torch.arange(count, device=device, dtype=dtype) * (2.0 * torch.pi / count)
        rings.append(torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=-1))
    return torch.cat(rings, dim=0)


def _get_foot_tip_positions_w(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Compute world-space 3D positions of the 4 foot tips from the terminal tibia links."""
    robot = env.scene[robot_cfg.name]
    if not hasattr(env, "_foot_body_ids"):
        body_names = robot.data.body_names
        missing = [name for name in FOOT_BODY_NAMES if name not in body_names]
        if missing:
            raise ValueError(f"Missing terminal leg bodies {missing}. Available: {body_names}")
        env._foot_body_ids = [body_names.index(name) for name in FOOT_BODY_NAMES]

    tibia_pos_w = robot.data.body_pos_w[:, env._foot_body_ids, :]
    tibia_quat_w = robot.data.body_quat_w[:, env._foot_body_ids, :]
    num_envs = tibia_pos_w.shape[0]
    num_feet = len(FOOT_BODY_NAMES)

    # Project from the knee/tibia origin down to the spherical foot tip
    tip_offset_local = torch.tensor([0.0, 0.0, -0.20], device=tibia_pos_w.device, dtype=tibia_pos_w.dtype)
    tip_offset_local = tip_offset_local.view(1, 1, 3).expand(num_envs, num_feet, 3)
    tip_offset_w = quat_apply(tibia_quat_w.reshape(-1, 4), tip_offset_local.reshape(-1, 3)).view(num_envs, num_feet, 3)
    return tibia_pos_w + tip_offset_w


def foot_height_sample_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    xy_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return world XYZ coordinates (N, 208, 3) for the 4x52 foot-centered scan points."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    foot_tips_w = _get_foot_tip_positions_w(env, robot_cfg)
    num_envs, num_feet, _ = foot_tips_w.shape

    local_offsets = _foot_scan_offsets(foot_tips_w.device, foot_tips_w.dtype)
    offsets_2d = local_offsets[None, None, :, :].expand(num_envs, num_feet, NUM_FOOT_SAMPLES, 2)
    if xy_noise is not None:
        offsets_2d = offsets_2d + xy_noise

    offsets_3d = torch.cat((offsets_2d, torch.zeros_like(offsets_2d[..., :1])), dim=-1)
    sample_quats = sensor.data.quat_w[:, None, None, :].expand(num_envs, num_feet, NUM_FOOT_SAMPLES, 4)
    rotated_offsets = quat_apply_yaw(sample_quats.reshape(-1, 4), offsets_3d.reshape(-1, 3)).view(
        num_envs, num_feet, NUM_FOOT_SAMPLES, 3
    )[..., :2]

    sample_xy_w = foot_tips_w[:, :, None, :2] + rotated_offsets
    sample_z_w = foot_tips_w[:, :, None, 2].expand(num_envs, num_feet, NUM_FOOT_SAMPLES)
    return torch.cat((sample_xy_w, sample_z_w.unsqueeze(-1)), dim=-1).view(num_envs, num_feet * NUM_FOOT_SAMPLES, 3)


def _sample_grid_at_world_xy(
    env: ManagerBasedRLEnv,
    sensor: RayCaster,
    world_z_grid: torch.Tensor,
    valid_grid: torch.Tensor,
    sample_xy_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample a (N, num_y, num_x) world-Z elevation grid at arbitrary (N, P, 2) world XY coordinates."""
    num_x, num_y, size_x, size_y, resolution = _height_grid_shape(sensor)
    ray_starts_w = sensor._ray_starts_w
    yaw_quat = sensor.data.quat_w
    yaw = torch.atan2(
        2.0 * (yaw_quat[:, 0] * yaw_quat[:, 3] + yaw_quat[:, 1] * yaw_quat[:, 2]),
        1.0 - 2.0 * (yaw_quat[:, 2].square() + yaw_quat[:, 3].square()),
    )
    cos_yaw = torch.cos(yaw)[:, None]
    sin_yaw = torch.sin(yaw)[:, None]

    start_delta = ray_starts_w[..., :2] - sensor.data.pos_w[:, None, :2]
    start_local_x = cos_yaw * start_delta[..., 0] + sin_yaw * start_delta[..., 1]
    start_local_y = -sin_yaw * start_delta[..., 0] + cos_yaw * start_delta[..., 1]
    origin_x = start_local_x.min(dim=1, keepdim=True).values
    origin_y = start_local_y.min(dim=1, keepdim=True).values

    delta_w = sample_xy_w - sensor.data.pos_w[:, None, :2]
    local_x = cos_yaw * delta_w[..., 0] + sin_yaw * delta_w[..., 1]
    local_y = -sin_yaw * delta_w[..., 0] + cos_yaw * delta_w[..., 1]

    grid_x = (2.0 * (local_x - origin_x) / (resolution * (num_x - 1))) - 1.0
    grid_y = (2.0 * (local_y - origin_y) / (resolution * (num_y - 1))) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)

    sampled_z = F.grid_sample(
        world_z_grid.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[:, 0, :, 0]
    sampled_valid = F.grid_sample(
        valid_grid.to(world_z_grid.dtype).unsqueeze(1),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0, :, 0] > 0.5
    return sampled_z, sampled_valid


def foot_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 0.35,
    include_obstacles: bool = True,
) -> torch.Tensor:
    """Noiseless 208-D (4x52) foot-centered height scan."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    num_x, num_y, _, _, _ = _height_grid_shape(sensor)
    ensure_obstacle_bounds_state(env)

    hits_w = sensor.data.ray_hits_w
    valid_grid = torch.isfinite(hits_w).all(dim=-1).view(env.num_envs, num_y, num_x)
    surface_z = torch.where(valid_grid.view(env.num_envs, -1), hits_w[..., 2], torch.zeros_like(hits_w[..., 2]))

    sample_pos_w = foot_height_sample_positions(env, robot_cfg, sensor_cfg)
    sample_xy_w = sample_pos_w[..., :2]

    sampled_z, sampled_valid = _sample_grid_at_world_xy(
        env, sensor, surface_z.view(env.num_envs, num_y, num_x), valid_grid, sample_xy_w
    )

    if include_obstacles:
        bounds = env._obstacle_bounds
        active = env._obstacle_active[:, None]
        dx = (sample_xy_w[..., 0] - bounds[:, 0, None]).abs()
        dy = (sample_xy_w[..., 1] - bounds[:, 1, None]).abs()
        on_box = active & (dx <= bounds[:, 2, None]) & (dy <= bounds[:, 3, None])
        sampled_z = torch.where(on_box, torch.maximum(sampled_z, bounds[:, 4, None]), sampled_z)

    heights = sensor.data.pos_w[:, 2, None] - sampled_z - offset
    heights = torch.where(sampled_valid, heights, torch.ones_like(heights))
    return torch.nan_to_num(heights, nan=1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def _ensure_student_noise_state(env: ManagerBasedRLEnv) -> None:
    if hasattr(env, "_episodic_foot_noise_xy"):
        return
    env._episodic_foot_noise_xy = torch.zeros((env.num_envs, 4, 1, 2), device=env.device)
    env._episodic_foot_noise_z = torch.zeros((env.num_envs, 4, 1), device=env.device)
    env._episodic_noise_mode = torch.zeros((env.num_envs, 1, 1), dtype=torch.long, device=env.device)


def reset_contact_belief_map(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """Reset per-episode contact belief map and resample episodic foot noise (w_x, w_y, w_z)."""
    sensor: RayCaster = env.scene.sensors["height_scanner"]
    _ensure_contact_belief_state(env, sensor)
    _ensure_student_noise_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    env._contact_belief_world_z[env_ids] = 0.0
    env._contact_belief_valid[env_ids] = False
    env._contact_belief_contact_known[env_ids] = False
    env._contact_belief_prev_origin[env_ids] = 0.0
    env._contact_belief_pose_valid[env_ids] = False
    env._contact_belief_last_step = -1

    # Select 60% nominal (0), 30% offset (1), 10% noisy (2) per episode
    n = env_ids.numel()
    r = torch.rand((n, 1, 1), device=env.device)
    modes = torch.where(r < 0.60, 0, torch.where(r < 0.90, 1, 2))
    env._episodic_noise_mode[env_ids] = modes

    # z6 (episodic xy std): nominal=0.05, offset=0.02, noisy=0.30; z7 (episodic z std): 0.10
    z6 = torch.where(modes == 0, 0.05, torch.where(modes == 1, 0.02, 0.30)).unsqueeze(-1)
    env._episodic_foot_noise_xy[env_ids] = torch.randn((n, 4, 1, 2), device=env.device) * z6
    env._episodic_foot_noise_z[env_ids] = torch.randn((n, 4, 1), device=env.device) * 0.10


def corrupted_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 0.35,
    noise_std: float = 0.005,
    dropout_prob: float = 0.03,
    blackout_prob: float = 0.02,
) -> torch.Tensor:
    """Height Sample Noise Model: per-point, per-foot, episodic XY/Z noise + outliers."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    num_x, num_y, _, _, _ = _height_grid_shape(sensor)
    _ensure_student_noise_state(env)

    # Clean mode for diagnostic visualization (--disable_scan_corruption)
    if noise_std <= 0.0 and dropout_prob <= 0.0 and blackout_prob <= 0.0:
        return foot_height_scan(env, sensor_cfg, robot_cfg, offset=offset, include_obstacles=False)

    num_envs = env.num_envs
    device = env.device
    modes = env._episodic_noise_mode  # (N, 1, 1): 0=nominal, 1=offset, 2=noisy

    z0 = 0.004
    z1 = torch.where(modes == 2, 0.10, noise_std)
    z2 = torch.where(modes == 2, 0.10, 0.01)
    z3 = torch.where(modes == 0, 0.04, torch.where(modes == 1, 0.10, 0.30))
    z4 = torch.where(modes == 0, 0.03, torch.where(modes == 1, 0.10, 0.30))
    z5 = torch.where(modes == 2, 0.30, dropout_prob)

    # XY perturbations: eps_px, eps_py ~ N(0, z0); eps_fx, eps_fy ~ N(0, z2); w_x, w_y ~ N(0, z6)
    eps_p_xy = torch.randn((num_envs, 4, NUM_FOOT_SAMPLES, 2), device=device) * z0
    eps_f_xy = torch.randn((num_envs, 4, 1, 2), device=device) * z2.unsqueeze(-1)
    total_xy_noise = eps_p_xy + eps_f_xy + env._episodic_foot_noise_xy

    # Sample ground-only terrain (invisible obstacles excluded from raw student scan)
    hits_w = sensor.data.ray_hits_w
    valid_grid = torch.isfinite(hits_w).all(dim=-1).view(num_envs, num_y, num_x)
    surface_z = torch.where(valid_grid.view(num_envs, -1), hits_w[..., 2], torch.zeros_like(hits_w[..., 2]))

    perturbed_pos_w = foot_height_sample_positions(env, robot_cfg, sensor_cfg, xy_noise=total_xy_w if False else total_xy_noise)
    sampled_z, sampled_valid = _sample_grid_at_world_xy(
        env, sensor, surface_z.view(num_envs, num_y, num_x), valid_grid, perturbed_pos_w[..., :2]
    )

    heights = (sensor.data.pos_w[:, 2, None] - sampled_z - offset).view(num_envs, 4, NUM_FOOT_SAMPLES)

    # Z perturbations: eps_pz ~ N(0, z1); eps_fz ~ N(0, z3); w_z ~ N(0, z7); outlier ~ N(0, z4) w.p. z5
    eps_pz = torch.randn((num_envs, 4, NUM_FOOT_SAMPLES), device=device) * z1
    eps_fz = torch.randn((num_envs, 4, 1), device=device) * z3
    outlier_mask = torch.rand((num_envs, 4, NUM_FOOT_SAMPLES), device=device) < z5
    eps_outlier = torch.where(
        outlier_mask,
        torch.randn((num_envs, 4, NUM_FOOT_SAMPLES), device=device) * z4,
        torch.zeros((num_envs, 4, NUM_FOOT_SAMPLES), device=device),
    )

    heights = heights + eps_pz + eps_fz + env._episodic_foot_noise_z + eps_outlier
    heights = heights.view(num_envs, 4 * NUM_FOOT_SAMPLES)
    heights = torch.where(sampled_valid, heights, torch.ones_like(heights))

    if blackout_prob > 0.0:
        blackout = torch.rand((num_envs, 1), device=device) < blackout_prob
        heights = torch.where(blackout, torch.ones_like(heights), heights)

    return torch.nan_to_num(heights, nan=1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def _ensure_contact_belief_state(env: ManagerBasedRLEnv, sensor: RayCaster) -> None:
    if hasattr(env, "_contact_belief_world_z"):
        return
    num_x, num_y, _, _, _ = _height_grid_shape(sensor)
    shape = (env.num_envs, num_x * num_y)
    env._contact_belief_world_z = torch.zeros(shape, device=env.device)
    env._contact_belief_valid = torch.zeros(shape, dtype=torch.bool, device=env.device)
    env._contact_belief_contact_known = torch.zeros(shape, dtype=torch.bool, device=env.device)
    env._contact_belief_prev_pos = torch.zeros((env.num_envs, 2), device=env.device)
    env._contact_belief_prev_origin = torch.zeros((env.num_envs, 2), device=env.device)
    env._contact_belief_prev_yaw = torch.zeros(env.num_envs, device=env.device)
    env._contact_belief_pose_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._contact_belief_last_step = -1
    env._cached_contact_foot_scan = torch.zeros((env.num_envs, 4 * NUM_FOOT_SAMPLES), device=env.device)


def contact_aware_foot_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 0.35,
) -> torch.Tensor:
    """Bilinearly sample the 4x52 foot scan points from the contact-corrected rolling terrain map."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    num_x, num_y, _, _, resolution = _height_grid_shape(sensor)
    _ensure_contact_belief_state(env, sensor)
    ensure_obstacle_bounds_state(env)

    step = int(env.common_step_counter)
    if env._contact_belief_last_step == step:
        return env._cached_contact_foot_scan

    ray_starts_w = sensor._ray_starts_w
    hits_w = sensor.data.ray_hits_w
    yaw_quat = sensor.data.quat_w
    yaw = torch.atan2(
        2.0 * (yaw_quat[:, 0] * yaw_quat[:, 3] + yaw_quat[:, 1] * yaw_quat[:, 2]),
        1.0 - 2.0 * (yaw_quat[:, 2].square() + yaw_quat[:, 3].square()),
    )

    start_delta = ray_starts_w[..., :2] - sensor.data.pos_w[:, None, :2]
    cos_yaw = torch.cos(yaw)[:, None]
    sin_yaw = torch.sin(yaw)[:, None]
    start_local_x = cos_yaw * start_delta[..., 0] + sin_yaw * start_delta[..., 1]
    start_local_y = -sin_yaw * start_delta[..., 0] + cos_yaw * start_delta[..., 1]
    current_origin = torch.stack(
        (start_local_x.min(dim=1).values, start_local_y.min(dim=1).values), dim=-1
    )

    xy_w = ray_starts_w[..., :2]
    delta = xy_w - env._contact_belief_prev_pos[:, None, :]
    prev_yaw = env._contact_belief_prev_yaw
    cos_prev = torch.cos(prev_yaw)[:, None]
    sin_prev = torch.sin(prev_yaw)[:, None]
    local_x = cos_prev * delta[..., 0] + sin_prev * delta[..., 1]
    local_y = -sin_prev * delta[..., 0] + cos_prev * delta[..., 1]
    inv_span_x = 2.0 / (resolution * (num_x - 1))
    inv_span_y = 2.0 / (resolution * (num_y - 1))
    grid_x = (local_x - env._contact_belief_prev_origin[:, 0, None]) * inv_span_x - 1.0
    grid_y = (local_y - env._contact_belief_prev_origin[:, 1, None]) * inv_span_y - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(env.num_envs, num_y, num_x, 2)

    packed_prev = torch.stack(
        (
            env._contact_belief_world_z,
            env._contact_belief_valid.to(env._contact_belief_world_z.dtype),
            env._contact_belief_contact_known.to(env._contact_belief_world_z.dtype),
        ),
        dim=1,
    ).reshape(env.num_envs, 3, num_y, num_x)

    warped_packed = F.grid_sample(
        packed_prev, grid, mode="nearest", padding_mode="zeros", align_corners=True
    ).reshape(env.num_envs, 3, -1)

    has_previous_pose = env._contact_belief_pose_valid[:, None]
    env._contact_belief_world_z = torch.where(has_previous_pose, warped_packed[:, 0], 0.0)
    env._contact_belief_valid = (warped_packed[:, 1] > 0.5) & has_previous_pose
    env._contact_belief_contact_known = (warped_packed[:, 2] > 0.5) & has_previous_pose

    scan_valid = torch.isfinite(hits_w).all(dim=-1)
    can_update_from_scan = scan_valid & ~env._contact_belief_contact_known
    env._contact_belief_world_z = torch.where(
        can_update_from_scan, hits_w[..., 2], env._contact_belief_world_z
    )
    env._contact_belief_valid |= scan_valid

    # Detect foot tip contact with either the training obstacle or a held-out obstacle in controll.py
    foot_tips_w = _get_foot_tip_positions_w(env, robot_cfg)
    bounds = env._obstacle_bounds
    active = env._obstacle_active[:, None]

    dx_f = (foot_tips_w[..., 0] - bounds[:, 0, None]).abs()
    dy_f = (foot_tips_w[..., 1] - bounds[:, 1, None]).abs()
    top_z = bounds[:, 4]
    foot_on_train_box = (
        active
        & (dx_f <= (bounds[:, 2, None] + 0.04))
        & (dy_f <= (bounds[:, 3, None] + 0.04))
        & (foot_tips_w[..., 2] <= (top_z[:, None] + 0.06))
    )

    # Also detect held-out invisible blocks in controll.py when a foot tip rests above the raw ground scan
    ground_at_feet, _ = _sample_grid_at_world_xy(
        env,
        sensor,
        hits_w[..., 2].view(env.num_envs, num_y, num_x),
        scan_valid.view(env.num_envs, num_y, num_x),
        foot_tips_w[..., :2],
    )
    robot = env.scene[robot_cfg.name]
    foot_vel_z = robot.data.body_lin_vel_w[:, env._foot_body_ids, 2].abs()
    foot_on_heldout_box = (
        (~active)
        & ((foot_tips_w[..., 2] - ground_at_feet) > 0.03)
        & ((foot_tips_w[..., 2] - ground_at_feet) < 0.25)
        & (foot_vel_z < 0.15)
    )

    foot_contact = foot_on_train_box | foot_on_heldout_box
    if foot_contact.any():
        contact_heights = torch.where(foot_on_train_box, top_z[:, None].expand_as(foot_tips_w[..., 2]), foot_tips_w[..., 2])
        ray_to_foot_sq = (xy_w[:, :, None, 0] - foot_tips_w[:, None, :, 0]).square() + (
            xy_w[:, :, None, 1] - foot_tips_w[:, None, :, 1]
        ).square()
        within_foot_radius = (ray_to_foot_sq <= (0.24**2)) & foot_contact[:, None, :]

        # If it's the training obstacle, clip to its exact box bounds; for held-out play props, stamp the 0.24m foot patch
        dx_r = (xy_w[..., 0] - bounds[:, 0, None]).abs()
        dy_r = (xy_w[..., 1] - bounds[:, 1, None]).abs()
        ray_in_train_box = (~active) | ((dx_r <= bounds[:, 2, None]) & (dy_r <= bounds[:, 3, None]))

        masked_heights = torch.where(
            within_foot_radius,
            contact_heights[:, None, :].expand_as(ray_to_foot_sq),
            torch.full_like(ray_to_foot_sq, -1e6),
        )
        stamped_z = masked_heights.max(dim=-1).values
        should_stamp = within_foot_radius.any(dim=-1) & ray_in_train_box

        env._contact_belief_world_z = torch.where(should_stamp, stamped_z, env._contact_belief_world_z)
        env._contact_belief_valid |= should_stamp
        env._contact_belief_contact_known |= should_stamp

    env._contact_belief_prev_pos = sensor.data.pos_w[:, :2].clone()
    env._contact_belief_prev_origin = current_origin.clone()
    env._contact_belief_prev_yaw = yaw.clone()
    env._contact_belief_pose_valid[:] = True
    env._contact_belief_last_step = step

    # Bilinearly sample the 4x52 foot scan points from the contact-updated rolling map
    sample_pos_w = foot_height_sample_positions(env, robot_cfg, sensor_cfg)
    sampled_z, sampled_valid = _sample_grid_at_world_xy(
        env,
        sensor,
        env._contact_belief_world_z.view(env.num_envs, num_y, num_x),
        env._contact_belief_valid.view(env.num_envs, num_y, num_x),
        sample_pos_w[..., :2],
    )
    heights = sensor.data.pos_w[:, 2, None] - sampled_z - offset
    heights = torch.where(sampled_valid, heights, torch.ones_like(heights))
    env._cached_contact_foot_scan = torch.nan_to_num(heights, nan=1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    return env._cached_contact_foot_scan