from .metaworld import MetaWorld
from .robodesk import RoboDesk
from .wrappers import NormalizeAction, TimeLimit, Async
import functools
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

def make_env(config:dict, mode):
        suite, task = config.task.split("_", 1)

        if suite == "metaworld":
            task = "-".join(task.split("_"))
            env = MetaWorld(
                task,
                config.seed,
                config.action_repeat,
                config.render_size,
                config.camera,
            )
            env = NormalizeAction(env)
        elif suite == "robodesk":
            env = RoboDesk(task, config.seed, config.action_repeat, config.render_size,
                                     evaluate=mode=='eval')
            env = NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        env = TimeLimit(env, config.time_limit)
        return env

make_async_env = lambda config, mode: Async(functools.partial(make_env, config, mode), config.envs_parallel)