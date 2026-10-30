from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from .quadrupedspider_base_env_cfg import QuadrupedspiderEnvCfg

@configclass
class TeacherObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        # Privileged states (Teacher PPO)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        clean_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class QuadrupedSpiderTeacherEnvCfg(QuadrupedspiderEnvCfg):
    observations: TeacherObservationsCfg = TeacherObservationsCfg()