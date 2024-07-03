import collections
import functools
import logging
import os
import re
import sys
import warnings
from pathlib import Path

logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml_package
yaml = yaml_package.YAML(typ='safe', pure=True)

import torch
import random

import wmlib
import wmlib.envs as envs
import wmlib.agents as agents
import wmlib.utils as utils
import wmlib.train as train




def main():

    configs = yaml.load(
        (Path(sys.argv[0]).parent.parent / "configs" / "dreamerv2.yaml").read_text()
    )
    parsed, remaining = utils.Flags(configs=["defaults"]).parse(known_only=True)
    config = utils.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = utils.Flags(config).parse(remaining)

    logdir = Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Set log directory:", logdir)

    assert torch.cuda.is_available(), 'No GPU found.'  
    assert config.precision in (16, 32), config.precision

    if config.precision == 16:
        print("setting fp16")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    if device != "cpu":
        torch.set_num_threads(1)

    # reproducibility
    utils.set_seed(config.seed)

    train_replay = wmlib.Replay(logdir / "train_episodes", seed=config.seed, **config.replay)
    eval_replay = wmlib.Replay(logdir / "eval_episodes", seed=config.seed, **dict(
        capacity=config.replay.capacity // 10,
        minlen=config.dataset.length,
        maxlen=config.dataset.length))
    
    step = utils.Counter(train_replay.stats["total_steps"])
    wandb_config = dict(config.wandb)
    task_name =  '-'.join(config.task.lower().split("_", 1))

    wandb_config['name']= f'{wandb_config["name"]}-{task_name}-seed{config.seed}'
    outputs = [
        utils.TerminalOutput(),
        utils.JSONLOutput(logdir),
        utils.WandbOutput(**wandb_config,config=dict(config))
    ]
    logger = utils.Logger(step, outputs, multiplier=config.action_repeat)

    
    # save experiment used config
    with open(logdir / "used_config.yaml", "w") as f:
        f.write("## command line input:\n## " + " ".join(sys.argv) + "\n##########\n\n")
        yaml.dump(dict(config), f)

    print("Create envs.")
    is_carla = config.task.split("_", 1)[0] == 'carla'
    num_eval_envs = min(config.envs, config.eval_eps)

    # only one env for carla
    if is_carla:
        assert config.envs == 1 and num_eval_envs == 1
    if config.envs_parallel == "none":
        train_envs = [envs.make_env(config, "train") for _ in range(config.envs)]
        eval_envs = [envs.make_env(config,"eval") for _ in range(num_eval_envs)]
    else:
        train_envs = [envs.make_async_env(config, "train") for _ in range(config.envs)]
        eval_envs = [envs.make_async_env(config, "eval") for _ in range(num_eval_envs)]

    agent = agents.DreamerV2(config, train_envs[0].obs_space, train_envs[0].act_space, step)
    trainer = train.Trainer(config, agent, train_replay, eval_replay, train_envs, eval_envs, step, logger)
    try:
        trainer.run(config.steps)
    except KeyboardInterrupt:
        print("Keyboard Interrupt - saving agent")
        trainer.save_agent(logdir / "variables.pt")
    except Exception as e:
        print("Training Error:", e)
        trainer.save_agent(logdir / "variables_error.pt")
        raise e
    finally:
        trainer.save_agent(logdir / "variables.pt")
        for env in train_envs + eval_envs:
            env.close()


if __name__ == "__main__":
    main()
