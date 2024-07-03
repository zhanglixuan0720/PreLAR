import numpy as np
import torch


class Driver:

    def __init__(self, envs, device, precision = 32, **kwargs):
        self._envs = envs
        self._device = device
        self._precision = precision
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.act_space for env in envs]
        self._dtype = torch.float16 if precision == 16 else torch.float32
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            # 1. reset check
            obs = {
                i: self._envs[i].reset()
                for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
            for i, ob in obs.items():
                assert not callable(ob)
                # self._obs[i] = ob() if callable(ob) else ob
                self._obs[i] = ob
                act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
                trans = {k: self._convert(v) for k, v in {**self._obs[i], **act}.items()}
                [fn(trans, worker=i, **self._kwargs) for fn in self._on_resets]
                self._eps[i] = [trans]

            # 2. observe
            obs = {k: torch.from_numpy(np.stack([o[k] for o in self._obs])).float() for k in self._obs[0]}  # convert before sending
            if len(obs['image'].shape) == 4:
                obs['image'] = obs['image'].permute(0, 3, 1, 2)

            #  this is a hack to make it work with the current policy  
            obs = {k: v.to(device=self._device, dtype=self._dtype) for k, v in obs.items()}

            # 3. policy
            actions, self._state = policy(obs, self._state, **self._kwargs)

            # 4. step
            actions = [{k: np.array(actions[k][i]) for k in actions} for i in range(len(self._envs))]
            assert len(actions) == len(self._envs)
            obs = [e.step(a) for e, a in zip(self._envs, actions)]
            # obs = [ob() if callable(ob) else ob for ob in obs]
            for i, (act, ob) in enumerate(zip(actions, obs)):
                # ob = _ob() if callable(_ob) else _ob
                assert not callable(ob)
                self._obs[i] = ob
                trans = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                [fn(trans, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(trans)
                step += 1
                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    [fn(ep, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
            # self._obs = obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
