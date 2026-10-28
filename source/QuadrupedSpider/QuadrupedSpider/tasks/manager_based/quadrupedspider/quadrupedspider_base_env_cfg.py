# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.sensors import RayCasterCfg, patterns
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp as custom_mdp

# =====================================================================================
# Robot definition
# =====================================================================================

QUADRUPED_SPIDER_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/rasmus/Robotics/MastersThesis/RLModel/QuadrupedSpider.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Diagonal yaw setup (+-45 degrees)
            "FRYawMotor": 0.7854,
            "RLYawMotor": 0.7854,
            "FLYawMotor": -0.7854,
            "RRYawMotor": -0.7854,

            # Hip pitch (angles thighs downward)
            ".*HipMotor": 0.35,

            # Knee bend (places feet firmly into the ground plane)
            ".*KneeMotor": -0.55,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*Motor"],
            effort_limit_sim=14.0,
            velocity_limit_sim=1890.0,
            stiffness=25.0,
            damping=1.2,
        ),
    },
)


# =====================================================================================
# Scene definition
# =====================================================================================


@configclass
class QuadrupedspiderSceneCfg(InteractiveSceneCfg):
    """Configuration for the quadruped spider robot scene."""

    # Procedural Rough Terrain (replaces GroundPlaneCfg)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
    )

    # Robot articulation
    robot: ArticulationCfg = QUADRUPED_SPIDER_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # LiDAR RayCaster sensor to scan the ground
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.1, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(size=(1.3, 1.3), resolution=0.13),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    tof_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.1, 0.0, 0.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(size=(0.0, 0.0), resolution=1.0),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],  # Bypasses the robot collision meshes
    )

    # Ambient lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

# =====================================================================================
# MDP settings
# =====================================================================================


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*Motor"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # IMU observations
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Target command from command manager
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # Joint relative positions and velocities
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)

        # Action history for smooth transitions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset root base (pose and orientation)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Reset joint positions with slight initial offset
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*Motor"]),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.5, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Survival status
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-3.0)

    # Velocity trackinhg
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )

    ## Posture and stability
    # ========================================================
    # Z axis movement
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Large action commands
    action_l2 = RewTerm(
        func=mdp.action_l2, 
        weight=-0.015
    )
    # Penalize joints exceeding the 5 Nm continuous rating
    continuous_torque_limit = RewTerm(
        func=custom_mdp.joint_effort_above_continuous_limit,
        weight=-0.005,
        params={"limit": 5.0, "asset_cfg": SceneEntityCfg("robot")}
    )
    # Penalize joint torque
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.00002,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # Penalize joints getting too close to their physical limits
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # IMU Gyro penalty: Penalize roll and pitch rotation rates
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )    
    # Smoothness regularization (Prevents jittery movement)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.04,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Drop termination
    base_height = DoneTerm(
        func=custom_mdp.tof_distance_below_minimum,  # Point this to your new function
        params={"minimum_distance": 0.18, "sensor_cfg": SceneEntityCfg("tof_sensor")},
    )

    # Roll/pitch threshold termination
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 2, "asset_cfg": SceneEntityCfg("robot")},
    )


# =====================================================================================
# Environment configuration
# =====================================================================================


@configclass
class QuadrupedspiderEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: QuadrupedspiderSceneCfg = QuadrupedspiderSceneCfg(num_envs=4096, env_spacing=4.0)

    # Sub-manager settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()

    # MDP reward and termination settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # Timings
        self.decimation = 4
        self.episode_length_s = 10.0

        # Viewport camera
        self.viewer.eye = (2.5, 2.5, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.05)

        # Simulation physics stepping (120 Hz PhysX / 30 Hz policy)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation