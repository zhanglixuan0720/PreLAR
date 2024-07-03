from ..core.driver import Driver
from ..core.replay import Replay
from .. import utils, agents
import numpy as np
import re
import collections
import torch
from pathlib import Path
from tqdm import tqdm

class Pretrainer:
    def __init__(self,config,agent:agents.BaseAgent,train_replay:Replay,eval_replay:Replay,step:utils.Counter,logger:utils.Logger) -> None:

        self.config = config
        
        
        self.device = config.device
        self._precision = config.precision
        self._dtype = torch.float16 if self._precision == 16 else torch.float32  # only on cuda
        self.metrics = collections.defaultdict(list)
        self.logdir = Path(config.logdir).expanduser()
        self.load_model_dir = Path(config.load_model_dir).expanduser()
        
        self.agent = agent
        self.step = step
        self.logger = logger

        self.train_replay = train_replay
        self.eval_replay = eval_replay

        self.should_log = utils.Every(config.log_every)
        self.should_video = utils.Every(config.video_every)
        self.should_save = utils.Every(config.eval_every)

        
        self.agent = self.agent.to(self.device)
        self.agent.init_optimizers()

        self.train_agent = CarryOverState(self.agent.train)

        self.train_dataset = iter(train_replay.dataset(**config.dataset))
        self.report_dataset = iter(train_replay.dataset(**config.dataset))
        if config.eval_video_list != 'none':
            self.eval_dataset = iter(eval_replay.dataset(**config.dataset))

        self.init_agent()
        self.load_agent(self.logdir / 'variables.pt')
        self.load_agent(self.load_model_dir / 'variables.pt')

    def run(self,steps):
        for _ in tqdm(range(int(steps)), total=int(steps), initial=int(self.step)):
            self.train_step()
                    
    def init_agent(self):
        print(f'Init agent from scratch. & Benchmark training.')
        self.agent.apply(self.weights_init)
        self.train_agent(self.next_batch(self.train_dataset))
        torch.cuda.empty_cache()

    def load_agent(self,path:Path):
        if path.exists():
            print(f'Load agent from checkpoint {path}.')
            self.agent.load_state_dict(torch.load(path))

    def save_agent(self,dir:Path,suffix=''):
        self.agent.save_model(dir, suffix)
    
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

    def train_step(self):
        train_metrics = self.train_agent(self.next_batch(self.train_dataset))
        [self.metrics[key].append(value) for key, value in train_metrics.items()]
        self.step.increment()
        
        if self.should_log(self.step):
            for name, values in self.metrics.items():
                self.logger.scalar(name, np.array(values, np.float64).mean())
                self.metrics[name].clear()
            # only video when log
            if self.should_video(self.step):
                report_metrics = self.agent.report(self.next_batch(self.report_dataset),recon=True)
                self.logger.add(report_metrics, prefix='train')
                self.logger.write(fps=True)
        
        if self.should_save(self.step):
            self.save_agent(self.logdir)
            if self.config.save_all_models and int(self.step) % 50000 == 1:
                self.save_agent(self.logdir, f'_s{int(self.step)}')
            
            if self.config.eval_video_list != 'none':
                eval_metrics = self.agent.eval(self.next_batch(self.eval_dataset))[1]
                for name, values in eval_metrics.items():
                    if name.endswith('loss'):
                        self.logger.scalar('eval/' + name, np.array(values, np.float64).mean())
                report_metrics =self.agent.report(self.next_batch(self.eval_dataset), recon=True)
                self.logger.add(report_metrics, prefix="val")
                self.logger.write(fps=True)
    
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

