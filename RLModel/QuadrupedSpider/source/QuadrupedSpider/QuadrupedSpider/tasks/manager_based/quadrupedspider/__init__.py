# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .teacher_env_cfg import QuadrupedSpiderTeacherEnvCfg
from .student_env_cfg import QuadrupedSpiderStudentEnvCfg


gym.register(
    id="Isaac-QuadrupedSpider-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadrupedSpiderTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-QuadrupedSpider-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadrupedSpiderStudentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_distill_cfg:QuadrupedSpiderDistillRunnerCfg",
    },
)