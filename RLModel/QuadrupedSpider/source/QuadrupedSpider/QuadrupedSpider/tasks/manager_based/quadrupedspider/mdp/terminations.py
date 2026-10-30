import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def tof_distance_below_minimum(
    env: ManagerBasedRLEnv, minimum_distance: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the ToF sensor reads below threshold or clips into the ground."""
    tof: RayCaster = env.scene[sensor_cfg.name]
    
    # Extract world positions
    hit_pos_w = tof.data.ray_hits_w[:, 0, :]
    sensor_pos_w = tof.data.pos_w
    
    # Calculate physical distance from sensor to the hit point
    distances = torch.norm(sensor_pos_w - hit_pos_w, dim=-1)
    
    # In Isaac Lab, a raycast miss defaults to [0.0, 0.0, 0.0]. 
    # If it misses, it means the sensor clipped under the floor mesh.
    missed = (hit_pos_w[:, 0] == 0.0) & (hit_pos_w[:, 1] == 0.0) & (hit_pos_w[:, 2] == 0.0)
    
    # Terminate if the ground is too close, OR if the belly clipped through the floor
    return (distances < minimum_distance) | missed