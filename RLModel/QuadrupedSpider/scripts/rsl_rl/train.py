# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as opr

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import QuadrupedSpider.tasks  # noqa: F401
from QuadrupedSpider.models import NativeDistillationRunner, TeacherActorCritic

opr.TeacherActorCritic = TeacherActorCritic

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def get_latest_teacher_checkpoint(expected_extero_dim: int) -> str:
    teacher_dir = os.path.abspath(os.path.join("logs", "rsl_rl", "QuadrupedSpider-PrivilegedTeacher"))
    if not os.path.exists(teacher_dir):
        raise FileNotFoundError(
            f"Privileged teacher directory not found at {teacher_dir}. "
            "Train Isaac-QuadrupedSpider-Teacher-v0 first."
        )

    run_dirs = [
        os.path.join(teacher_dir, d)
        for d in os.listdir(teacher_dir)
        if os.path.isdir(os.path.join(teacher_dir, d))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found inside: {teacher_dir}")

    def extract_iter(p):
        base = os.path.basename(p)
        num = "".join(filter(str.isdigit, base))
        return int(num) if num else os.path.getmtime(p)

    for run_dir in sorted(run_dirs, key=os.path.getmtime, reverse=True):
        checkpoints = glob.glob(os.path.join(run_dir, "**", "model_*.pt"), recursive=True)
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(run_dir, "*.pt"))
        for checkpoint_path in sorted(checkpoints, key=extract_iter, reverse=True):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            extero_w = state_dict.get("extero_encoder.net.0.weight")
            if extero_w is not None and extero_w.shape[1] * 4 == expected_extero_dim:
                return checkpoint_path

    raise FileNotFoundError(
        f"No TeacherActorCritic checkpoint with extero_dim={expected_extero_dim} found under "
        f"{teacher_dir}. Train the teacher task with the current observation config."
    )


def load_teacher_policy(checkpoint_path: str, obs: dict, num_actions: int, device: str) -> TeacherActorCritic:
    """Load a TeacherActorCritic using its saved network config."""
    run_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(run_dir) == "exported":
        run_dir = os.path.dirname(run_dir)
    agent_cfg_path = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(agent_cfg_path):
        raise FileNotFoundError(
            f"Teacher config not found at {agent_cfg_path}. Select a checkpoint from a saved training run."
        )

    with open(agent_cfg_path, encoding="utf-8") as agent_cfg_file:
        teacher_cfg = yaml.safe_load(agent_cfg_file)
    policy_cfg = teacher_cfg.get("policy", {})
    if not isinstance(policy_cfg, dict):
        raise ValueError(f"Teacher policy config in {agent_cfg_path} is missing or invalid.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    teacher_obs_groups = ["proprioception", "exteroception", "privileged"]
    teacher_policy = TeacherActorCritic(
        obs=obs,
        obs_groups={
            "policy": teacher_obs_groups,
            "critic": teacher_obs_groups,
        },
        num_actions=num_actions,
        actor_hidden_dims=policy_cfg.get("actor_hidden_dims", [256, 160, 128]),
        critic_hidden_dims=policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
        activation=policy_cfg.get("activation", "leaky_relu"),
        actor_obs_normalization=policy_cfg.get("actor_obs_normalization", True),
        critic_obs_normalization=policy_cfg.get("critic_obs_normalization", True),
    ).to(device)
    teacher_policy.load_state_dict(state_dict)
    teacher_policy.eval()
    teacher_policy.requires_grad_(False)
    print(
        f"[INFO]: Teacher architecture loaded from {agent_cfg_path}, "
        f"using observation groups {teacher_obs_groups}."
    )
    return teacher_policy


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if hasattr(args_cli, "checkpoint") and args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    elif agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = None

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        if agent_cfg.resume and resume_path:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path, load_optimizer=True)

    elif agent_cfg.class_name == "DistillationRunner":
        teacher_checkpoint_path = getattr(agent_cfg, "teacher_checkpoint", None)
        obs = env.get_observations()
        expected_extero_dim = int(obs["exteroception"].shape[-1])
        if not teacher_checkpoint_path:
            teacher_checkpoint_path = get_latest_teacher_checkpoint(expected_extero_dim)

        print(f"[INFO]: Dynamically loading frozen teacher checkpoint from: {teacher_checkpoint_path}")

        teacher_policy = load_teacher_policy(
            checkpoint_path=teacher_checkpoint_path,
            obs=obs,
            num_actions=env.num_actions,
            device=agent_cfg.device,
        )

        runner = NativeDistillationRunner(
            env=env,
            train_cfg=agent_cfg.to_dict(),
            teacher_policy=teacher_policy,
            log_dir=log_dir,
            device=agent_cfg.device,
        )
        if agent_cfg.resume and resume_path:
            print(f"[INFO]: Resuming student checkpoint from: {resume_path}")
            runner.load(resume_path, load_optimizer=True)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.add_git_repo_to_log(__file__)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()