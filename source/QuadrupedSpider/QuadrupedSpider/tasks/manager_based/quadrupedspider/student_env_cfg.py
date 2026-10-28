from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from .quadrupedspider_base_env_cfg import QuadrupedspiderEnvCfg, EventCfg
from . import mdp as custom_mdp


@configclass
class StudentObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Student observations: Noisy proprioception with history for belief estimation."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 20

    @configclass
    class TeacherCfg(ObsGroup):
        """Teacher observations: Exact terms, ordering, and clean states matching Teacher PPO."""
        # Clean proprioception
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        # Privileged states
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        clean_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class StudentEventCfg(EventCfg):
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 1.5),
            "operation": "add",
        },
    )


@configclass
class QuadrupedSpiderStudentEnvCfg(QuadrupedspiderEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()
    events: StudentEventCfg = StudentEventCfg()