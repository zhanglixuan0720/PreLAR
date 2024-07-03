import os

import gym
import numpy as np


class RoboDesk:

    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), camera=None, use_gripper=False, evaluate=False):
        import robodesk
  

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}"
        reward_type = 'success' if evaluate else 'dense'
        self._evaluate = evaluate
        self._env = robodesk.RoboDesk(task=task, reward=reward_type, action_repeat=action_repeat, episode_length=500, image_size=size[0])
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
            "state": self._env.observation_space["qpos_robot"],
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        # if self._use_gripper:
        #     spaces["gripper_image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
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
            if self._evaluate:
                success = rew #float(info["success"])
            else:
                success = 0
            reward += rew or 0.0
            if done or success == 1.0:
                break
        assert success in [0.0, 1.0]
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": state["image"],
            "state": state["qpos_robot"],
            "success": success,
        }
        # if self._use_gripper:
        #     obs["gripper_image"] = self._env.sim.render(
        #         *self._size, mode="offscreen", camera_name="behindGripper"
        #     )
        return obs

    def reset(self):
        # if self._camera == "corner2":
        #     self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": state["image"],
            "state": state["qpos_robot"],
            "success": False,
        }
        # if self._use_gripper:
        #     obs["gripper_image"] = self._env.sim.render(
        #         *self._size, mode="offscreen", camera_name="behindGripper"
        #     )
        return obs

    def close(self):
        ...

if __name__ == '__main__':
    robo_desk_env = RoboDesk('open_slide',0,1,(64,64),evaluate=False)
    obs = robo_desk_env.reset()
    done = False
    while not done:
        action = robo_desk_env.act_space["action"].sample()
        action = {"action":action}
        obs = robo_desk_env.step(action)