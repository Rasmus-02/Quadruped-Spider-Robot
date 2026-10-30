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
        """Proprioceptive measurements (45 dimensions)."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0))
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class ExteroceptionCfg(ObsGroup):
        """Corrupted height scan for the student belief encoder (121 dimensions)."""
        noisy_height_scan = ObsTerm(
            func=custom_mdp.corrupted_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "noise_std": 0.01,      # Gaussian noise with std dev of x meters
                "dropout_prob": 0.2,   # % chance per ray to be dropped
                "blackout_prob": 0.0,  # % chance per step to be completely blind
            },
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Clean height scan and linear velocity for distillation targets (124 dimensions)."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        clean_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    # Group assignments for Isaac Lab manager and RSL-RL
    policy: ProprioceptionCfg = ProprioceptionCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    exteroception: ExteroceptionCfg = ExteroceptionCfg()
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