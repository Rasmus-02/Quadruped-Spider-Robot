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
    class ProprioceptionCfg(ObsGroup):
        """Proprioceptive inputs (57 dims)."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.05, clip=(-100.0, 100.0))
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class ExteroceptionCfg(ObsGroup):
        """corrupted 4x52 foot-centered height scan (208 dims)."""
        noisy_height_grid = ObsTerm(
            func=custom_mdp.corrupted_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "robot_cfg": SceneEntityCfg("robot"),
                "noise_std": 0.005,
                "dropout_prob": 0.03,
                "blackout_prob": 0.02,
            },
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class TeacherExteroceptionCfg(ObsGroup):
        """Contact-corrected 4x52 foot height scan (208 dims) used as reconstruction target."""
        contact_corrected_height_grid = ObsTerm(
            func=custom_mdp.contact_aware_foot_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Privileged physical state (base_lin_vel)."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    exteroception: ExteroceptionCfg = ExteroceptionCfg()
    teacher_exteroception: TeacherExteroceptionCfg = TeacherExteroceptionCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


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