# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play an RL agent using Isaac Sim's internal ROS 2 bridge for Xbox teleoperation."""

import argparse
import os
import sys
import time

# Purge system ROS 2 (3.10) paths from environment so they don't break Python 3.11
sys.path = [p for p in sys.path if not p.startswith("/opt/ros")]
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = ":".join(
        [p for p in os.environ["PYTHONPATH"].split(":") if not p.startswith("/opt/ros")]
    )

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and ROS 2 teleop.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Enable Built-in Isaac Sim ROS 2 Bridge (compiled for internal Python)"""
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

"""Imports that depend on Isaac Sim's internal ROS 2 environment"""
import gymnasium as gym
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import QuadrupedSpider.tasks  # noqa: F401


class RosTeleopNode(Node):
    """ROS 2 listener node for teleoperation and manual reset commands."""

    def __init__(self):
        super().__init__("isaac_teleop_node")
        self.cmd_vel = [0.0, 0.0, 0.0]
        self.reset_requested = False

        self.twist_sub = self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 10)
        self.joy_sub = self.create_subscription(Joy, "/joy", self._joy_callback, 10)

    def _cmd_vel_callback(self, msg: Twist):
        self.cmd_vel = [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)]

    def _joy_callback(self, msg: Joy):
        # Button index 0 is 'A' on standard Xbox mappings
        if len(msg.buttons) > 0 and msg.buttons[0] == 1:
            self.reset_requested = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent using ROS 2 teleoperation."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Keep robot alive indefinitely during play
    env_cfg.episode_length_s = 1e6
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None

    # Locate checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)

    # Initialize Environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ROS 2 node initialization
    if not rclpy.ok():
        rclpy.init()
    ros_node = RosTeleopNode()

    print("\n" + "=" * 65)
    print("[INFO] Isaac Sim ROS 2 Teleop Listener Active:")
    print("  - Listening on /cmd_vel (Twist)")
    print("  - Listening on /joy (Button 0 'A' triggers reset)")
    print("=" * 65 + "\n")

    command_term = None
    if isinstance(env.unwrapped, ManagerBasedRLEnv) and "base_velocity" in env.unwrapped.command_manager.active_terms:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    loop_step = 0

    while simulation_app.is_running():
        start_time = time.time()

        # Spin ROS events non-blockingly
        rclpy.spin_once(ros_node, timeout_sec=0.0)

        with torch.inference_mode():
            # Manual reset triggered by Xbox button A
            if ros_node.reset_requested:
                print("[TELEOP] Button A pressed -> Resetting robot!")
                ros_node.reset_requested = False
                all_env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
                env.unwrapped.reset(env_ids=all_env_ids)
                obs = env.get_observations()

            # Push velocity command into Isaac Lab manager buffer
            cmd_tensor = torch.tensor(ros_node.cmd_vel, dtype=torch.float32, device=env.unwrapped.device)
            if command_term is not None:
                command_term.command[:] = cmd_tensor.repeat(env.unwrapped.num_envs, 1)

            # Step policy & physics
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # Re-apply command tensor immediately after step
            if command_term is not None:
                command_term.command[:] = cmd_tensor.repeat(env.unwrapped.num_envs, 1)

            if hasattr(policy, "reset"):
                policy.reset(dones)

            if loop_step % 60 == 0:
                print(
                    f"[Step {loop_step}] /cmd_vel: [{ros_node.cmd_vel[0]:.2f}, {ros_node.cmd_vel[1]:.2f}, {ros_node.cmd_vel[2]:.2f}] | Action Norm: {torch.norm(actions).item():.3f}"
                )

        loop_step += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    ros_node.destroy_node()
    rclpy.shutdown()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()