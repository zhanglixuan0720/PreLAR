import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
from pathlib import Path


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml_package
yaml = yaml_package.YAML(typ='safe', pure=True)
import torch
import random

import wmlib
import wmlib.envs as envs
import wmlib.agents as agents
import wmlib.utils as utils
import wmlib.datasets as datasets
import wmlib.train as train


def main():

    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent.parent / "configs" / "apv_pretraining.yaml").read_text()
    )
    parsed, remaining = utils.Flags(configs=["defaults"]).parse(known_only=True)
    config = utils.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = utils.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser() # expand the user's home directory, e.g. ~/logs to /home/user/logs
    load_logdir = pathlib.Path(config.load_logdir).expanduser()
    load_model_dir = pathlib.Path(config.load_model_dir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)
    print("Loading Logdir", load_logdir)

    assert torch.cuda.is_available(), 'No GPU found.'  
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        print("setting fp16")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    if device != "cpu":
        torch.set_num_threads(1)

    # reproducibility
    utils.set_seed(config.seed)
    train_replay = datasets.make_action_free_dataset(config['dataset_type'],config['video_dirs'],config['video_lists'],config['replay']['minlen'],config['manual_labels'],seed=config.seed,**config.replay)
    eval_replay = None
    if config.eval_video_list != 'none':
        eval_replay = datasets.make_action_free_dataset(config['dataset_type'],config['video_dirs'],config['eval_video_list'],config['replay']['minlen'],config['manual_labels'],seed=config.seed,**config.replay)

    step = utils.Counter(train_replay.stats["total_steps"])
    wandb_config = dict(config.wandb)
    wandb_config['name']= f'{wandb_config["name"]}-{config["dataset_type"]}-seed{config.seed}'
    step = utils.Counter(train_replay.stats["total_steps"])
    outputs = [
        utils.TerminalOutput(),
        utils.JSONLOutput(logdir),
        utils.WandbOutput(**wandb_config,config=dict(config))
    ]
    logger = utils.Logger(step, outputs, multiplier=config.action_repeat)

    print("Create envs.")
    env = envs.make_env(config, 'train')
    act_space, obs_space = env.act_space, env.obs_space

    agent = agents.APV_Pretrain(config, obs_space, act_space, step)
    pretrainer = train.Pretrainer(config,agent,train_replay,eval_replay,step,logger)
    pretrainer.run(config.steps)
    pretrainer.save_agent(logdir)
    env.close()


if __name__ == "__main__":
    main()
