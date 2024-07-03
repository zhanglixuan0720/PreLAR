import torch
from tqdm import tqdm
from pathlib import Path

from wmlib.core.replay import Replay
from .trainer import Trainer
from wmlib.utils import Counter, Logger
from wmlib.agents import BaseAgent

class Finetuner(Trainer):
    def __init__(self, config, agent: BaseAgent, train_replay: Replay, eval_replay: Replay, train_envs, eval_envs, step: Counter, logger: Logger) -> None:
        super().__init__(config, agent, train_replay, eval_replay, train_envs, eval_envs, step, logger)

    def load_agent(self,path:Path,iteration:int=0):
        if path.exists():
            print(f'Load agent from checkpoint {path}.')
            self.agent.load_state_dict(torch.load(path))
        else:
            load_logdir = Path(self.config.load_logdir).expanduser()
            if load_logdir != 'none':
                if 'af_rssm' in self.config.load_modules:
                    print(self.agent.wm.af_rssm.load_state_dict(torch.load(load_logdir / 'rssm_variables.pt'), strict=self.config.load_strict))
                    print(f'Load af_rssm from checkpoint {load_logdir}/rssm_variables.pt.')
                if 'encoder' in self.config.load_modules:
                    print(self.agent.wm.encoder.load_state_dict(torch.load(
                        load_logdir / 'encoder_variables.pt'), strict=self.config.load_strict))
                    print(f'Load encoder from checkpoint {load_logdir}/encoder_variables.pt.')
                if 'decoder' in self.config.load_modules:
                    print(self.agent.wm.heads['decoder'].load_state_dict(torch.load(
                        load_logdir / 'decoder_variables.pt'), strict=self.config.load_strict))
                    print(f'Load decoder from checkpoint {load_logdir}/decoder_variables.pt.')
            print(f'Pretrain agent from scratch {iteration} iterations.')
            for _ in tqdm(range(iteration)):
                self.train_agent(self.next_batch(self.train_dataset))