from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from .quadrupedspider_base_env_cfg import QuadrupedspiderEnvCfg
from . import mdp as custom_mdp


@configclass
class TeacherObservationsCfg:
    @configclass
    class ProprioceptionCfg(ObsGroup):
        """Proprioceptive inputs shared with the student (57 dims)."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.05, clip=(-100.0, 100.0))
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class ExteroceptionCfg(ObsGroup):
        """Noiseless 4x52 concentric foot-centered height scan (208 dims)."""
        true_foot_scan = ObsTerm(
            func=custom_mdp.foot_height_scan,
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
        """Privileged physical state for the teacher privileged encoder."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    exteroception: ExteroceptionCfg = ExteroceptionCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class QuadrupedSpiderTeacherEnvCfg(QuadrupedspiderEnvCfg):
    observations: TeacherObservationsCfg = TeacherObservationsCfg()