import os

import gym
import numpy as np


class MetaWorld:

    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), camera=None, use_gripper=False):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._use_gripper = use_gripper

        self._camera = camera

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        if self._use_gripper:
            spaces["gripper_image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action["action"])
            success = float(info["success"])
            reward += rew or 0.0
            if done or success == 1.0:
                break
        assert success in [0.0, 1.0]
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": success,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": False,
        }
        if self._use_gripper:
            obs["gripper_image"] = self._env.sim.render(
                *self._size, mode="offscreen", camera_name="behindGripper"
            )
        return obs

    def close(self):
        ...