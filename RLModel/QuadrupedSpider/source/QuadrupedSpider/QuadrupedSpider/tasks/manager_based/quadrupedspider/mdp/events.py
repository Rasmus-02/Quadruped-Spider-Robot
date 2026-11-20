import torch

OBSTACLE_BOX_SIZE = (0.36, 0.46, 0.15)
EXPOSED_HEIGHTS = (0.05, 0.10, 0.15)


def ensure_obstacle_bounds_state(env) -> None:
    if hasattr(env, "_obstacle_bounds"):
        return
    # Columns: [center_x, center_y, half_x, half_y, top_z]
    env._obstacle_bounds = torch.zeros((env.num_envs, 5), device=env.device)
    env._obstacle_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def randomize_hidden_belief_obstacles(
    env,
    env_ids: torch.Tensor,
    active_probability: float = 0.5,
    x_range: tuple[float, float] = (0.45, 1.0),
    y_range: tuple[float, float] = (-0.3, 0.3),
) -> None:
    """Place the single reusable collision obstacle flush with the local ground surface."""
    ensure_obstacle_bounds_state(env)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    origins = env.scene.env_origins[env_ids]
    active = torch.rand(env_ids.numel(), device=env.device) < active_probability
    variants = torch.randint(len(EXPOSED_HEIGHTS), (env_ids.numel(),), device=env.device)
    height_choices = torch.tensor(EXPOSED_HEIGHTS, device=env.device, dtype=origins.dtype)
    exposed_h = height_choices[variants]

    x_offset = torch.empty(env_ids.numel(), device=env.device).uniform_(*x_range)
    y_offset = torch.empty(env_ids.numel(), device=env.device).uniform_(*y_range)
    target_x = origins[:, 0] + x_offset
    target_y = origins[:, 1] + y_offset

    # Estimate local terrain surface Z from the height scanner if available, else env_origins Z
    ground_z = origins[:, 2].clone()
    if "height_scanner" in env.scene.sensors:
        scanner = env.scene.sensors["height_scanner"]
        hits_w = scanner.data.ray_hits_w[env_ids]
        valid_hits = torch.isfinite(hits_w).all(dim=-1)
        if valid_hits.any():
            dist_sq = (hits_w[..., 0] - target_x[:, None]).square() + (
                hits_w[..., 1] - target_y[:, None]
            ).square()
            dist_sq = torch.where(valid_hits, dist_sq, torch.full_like(dist_sq, 1e6))
            nearest_idx = dist_sq.argmin(dim=-1, keepdim=True)
            sampled_z = torch.gather(hits_w[..., 2], 1, nearest_idx).squeeze(-1)
            has_valid = valid_hits.any(dim=-1)
            ground_z = torch.where(has_valid, sampled_z, ground_z)

    top_z = ground_z + exposed_h
    box_half_z = OBSTACLE_BOX_SIZE[2] * 0.5

    # Sink the invisible block so its top face sits at ground_z + exposed_h
    active_center_z = top_z - box_half_z
    inactive_center_z = origins[:, 2] - 2.0

    pos_x = torch.where(active, target_x, origins[:, 0])
    pos_y = torch.where(active, target_y, origins[:, 1])
    pos_z = torch.where(active, active_center_z, inactive_center_z)

    if "belief_obstacle" in env.scene.keys():
        asset = env.scene["belief_obstacle"]
        positions = torch.stack((pos_x, pos_y, pos_z), dim=-1)
        orientations = torch.zeros((env_ids.numel(), 4), device=env.device, dtype=origins.dtype)
        orientations[:, 0] = 1.0
        asset.write_root_pose_to_sim(
            torch.cat((positions, orientations), dim=-1),
            env_ids=env_ids,
        )

    bounds = torch.zeros((env_ids.numel(), 5), device=env.device, dtype=origins.dtype)
    bounds[:, 0] = pos_x
    bounds[:, 1] = pos_y
    bounds[:, 2] = OBSTACLE_BOX_SIZE[0] * 0.5
    bounds[:, 3] = OBSTACLE_BOX_SIZE[1] * 0.5
    bounds[:, 4] = top_z

    env._obstacle_bounds[env_ids] = bounds
    env._obstacle_active[env_ids] = active