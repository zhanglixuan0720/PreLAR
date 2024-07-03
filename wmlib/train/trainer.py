from ..core.driver import Driver
from ..core.replay import Replay
from .. import utils, agents
import numpy as np
import re
import collections
import torch
from pathlib import Path
from tqdm import tqdm

class Trainer:
    def __init__(self,config,agent:agents.BaseAgent,train_replay:Replay,eval_replay:Replay,train_envs,eval_envs,step:utils.Counter,logger:utils.Logger) -> None:

        self.config = config
        
        self.device = config.device
        self._precision = config.precision
        self._dtype = torch.float16 if self._precision == 16 else torch.float32  # only on cuda
        self.metrics = collections.defaultdict(list)
        self.logdir = Path(config.logdir).expanduser()
        
        self.agent = agent
        self.step = step
        self.logger = logger

        self.train_replay = train_replay
        self.eval_replay = eval_replay

        self.train_driver = Driver(train_envs,self.device, precision=self._precision)
        self.train_driver.on_episode(lambda ep: self.per_episode(ep, mode="train"))
        self.train_driver.on_step(lambda tran, worker: step.increment())
        self.train_driver.on_step(train_replay.add_step)
        self.train_driver.on_reset(train_replay.add_step)

        self.eval_driver = Driver(eval_envs, config.device, precision=self._precision)
        self.eval_driver.on_episode(lambda ep: self.per_episode(ep, mode="eval"))
        self.eval_driver.on_episode(eval_replay.add_episode)

        self.should_train = utils.Every(config.train_every)
        self.should_log = utils.Every(config.log_every)
        self.should_video_train = utils.Every(config.eval_every)
        self.should_video_eval = utils.Every(config.eval_every)
        self.should_expl = utils.Until(config.expl_until // config.action_repeat)

        prefill_steps = max(0, config.prefill - int(step))
        print(f"Prefill dataset with {prefill_steps} steps.")
        if prefill_steps:
            random_agent = agents.RandomAgent(train_envs[0].act_space)
            self.train_driver(random_agent, steps=prefill_steps, episodes=1)
            self.eval_driver(random_agent, episodes=1)
            self.train_driver.reset()
            self.eval_driver.reset()
        
        self.agent = self.agent.to(self.device)
        self.agent.init_optimizers()

        self.train_agent = CarryOverState(self.agent.train)
        self.train_driver.on_step(self.train_step)

        self.train_dataset = iter(train_replay.dataset(**config.dataset))
        self.report_dataset = iter(train_replay.dataset(**config.dataset))
        self.eval_dataset = iter(eval_replay.dataset(pin_memory=False, **config.dataset))

        self.init_agent()
        self.load_agent(self.logdir / 'variables.pt',iteration=self.config.pretrain)

        self.train_policy = lambda *args: self.agent.policy(
            *args, mode="explore" if self.should_expl(step) else "train")
        self.eval_policy = lambda *args: self.agent.policy(*args, mode="eval")

    def run(self,steps, stop_steps=-1):
        while self.step < steps:
            self.logger.write()
            print(f"Step {self.step}: start evaluation.")
            report_metrics = self.agent.report(self.next_batch(self.eval_dataset))
            self.logger.add(report_metrics, prefix="eval")
            self.eval_driver(self.eval_policy, episodes=self.config.eval_eps)
            self.logger.write()  # aggregate eval results
            is_carla = self.config.task.split("_", 1)[0] == 'carla'
            if is_carla and int(self.step) % 50000 < 5000:
                torch.save(self.agent.state_dict(), self.logdir / ("variables_s" + str(int(self.step)) + ".pt"))
            if stop_steps != -1 and self.step >= stop_steps:
                break
            print(f"Step {self.step}: start training.")
            self.train_driver(self.train_policy, steps=self.config.eval_every)
            torch.save(self.agent.state_dict(), self.logdir / "variables.pt")

    def init_agent(self):
        print(f'Init agent from scratch. & Benchmark training.')
        self.agent.apply(self.weights_init)
        self.train_agent(self.next_batch(self.train_dataset))
        torch.cuda.empty_cache()

    def load_agent(self,path:Path,iteration:int=0):
        if path.exists():
            print(f'Load agent from checkpoint {path}.')
            self.agent.load_state_dict(torch.load(path))
        else:
            print(f'Pretrain agent from scratch {iteration} iterations.')
            for _ in tqdm(range(iteration)):
                self.train_agent(self.next_batch(self.train_dataset))

    def save_agent(self,path:Path):
        print(f'Save agent to checkpoint {path}.')
        torch.save(self.agent.state_dict(), path)
    
    def weights_init(self, m):
        if hasattr(m,'original_name'):
            classname = m.original_name
        else:
            classname = m.__class__.__name__
        if classname.find('LayerNorm') == -1 and classname.find('BatchNorm') == -1 and hasattr(m, "weight"):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None and m.bias.data is not None:
                torch.nn.init.zeros_(m.bias)
        if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
            print("setting memory format to channels last")
            m.to(memory_format=torch.channels_last)

    def train_step(self, tran, worker):
        if self.should_train(self.step):
            for _ in range(self.config.train_steps):
                train_metrics = self.train_agent(self.next_batch(self.train_dataset))
                [self.metrics[key].append(value) for key, value in train_metrics.items()]
        if self.should_log(self.step):
            for name, values in self.metrics.items():
                self.logger.scalar(name, np.array(values, np.float64).mean())
                self.metrics[name].clear()
            report_metrics = self.agent.report(self.next_batch(self.report_dataset))
            self.logger.add(report_metrics, prefix="train")
            self.logger.write(fps=True)

    def per_episode(self, ep, mode):
        episode_length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        if "metaworld" in self.config.task or 'rlbench' in self.config.task or 'robodesk' in self.config.task :
            success = float(np.any(ep['success'])) # float(np.sum(ep["success"]) >= 1.0)
            print(f"{mode.title()} episode has {success} success, {episode_length} steps and return {score:.1f}.")
            self.logger.scalar(f"{mode}/episode_success", success)
        elif "carla" in self.config.task:
            print(ep["dist_s"].shape, ep["collision_cost"].shape, ep["steer_cost"].shape)
            dist_s = float(np.max(ep["dist_s"]))
            collision_cost = float(np.min(ep["collision_cost"]))
            steer_cost = float(np.min(ep["steer_cost"]))
            centering_cost = float(np.min(ep["centering_cost"]))
            print(f"{mode.title()} episode max dist_s is {dist_s}, {episode_length} steps and return {score:.1f}.")
            print(f"{mode.title()} episode has {collision_cost} collision cost, {steer_cost} steer cost and {centering_cost} centering cost.")
            self.logger.scalar(f"{mode}/episode_max_dist_s", dist_s)
            self.logger.scalar(f"{mode}/episode_collision_cost", collision_cost)
            self.logger.scalar(f"{mode}/episode_steer_cost", steer_cost)
            self.logger.scalar(f"{mode}/episode_centering_cost", centering_cost)
        else:
            print(f"{mode.title()} episode has {episode_length} steps and return {score:.1f}.")
        self.logger.scalar(f"{mode}/episode_return", score)
        self.logger.scalar(f"{mode}/episode_length", episode_length)
        for key, value in ep.items():
            if re.match(self.config.log_keys_sum, key):
                self.logger.scalar(f"{mode}/sum/{key}", value.sum())
            if re.match(self.config.log_keys_mean, key):
                self.logger.scalar(f"{mode}/sum/{key}", value.mean())
            if re.match(self.config.log_keys_max, key):
                self.logger.scalar(f"{mode}/sum/{key}", value.max(0).mean())
        should = {"train": self.should_video_train, "eval": self.should_video_eval}[mode]
        if should(self.step):
            for key in self.config.log_keys_video:
                self.logger.video(f"{mode}/episode_{key}", ep[key].transpose(0,3,1,2)) # policy
        replay = dict(train=self.train_replay, eval=self.eval_replay)[mode]
        self.logger.add(replay.stats, prefix=mode)
        if mode != 'eval':  # NOTE: to aggregate eval results at last
            self.logger.write()
    
    def next_batch(self, iter):
        # casts to fp16 and cuda
        out = {k: v.to(device=self.device, dtype=self._dtype) for k, v in next(iter).items()}
        return out
    

class CarryOverState:

    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        return out 

