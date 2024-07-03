import json
import time
import collections
import numpy as np
import wandb
from pathlib import Path
from .counter import Counter

class Logger:

    def __init__(self, step:Counter, outputs:list, multiplier=1):
        self._step = step
        self._outputs = outputs
        self._multiplier = multiplier
        self._last_step = None
        self._last_time = None
        self._metrics = []

    def add(self, mapping, prefix=None):
        step = int(self._step) * self._multiplier
        for name, value in dict(mapping).items():
            name = f'{prefix}/{name}' if prefix else name
            value = np.array(value)
            assert len(value.shape) in (0, 2, 3, 4), f'Shape {value.shape} for {name} cannot be interpreted as scalar, image, or video.'
            self._metrics.append((step, name, value))

    def scalar(self, name, value):
        self.add({name: value})

    def image(self, name, value):
        self.add({name: value})

    def video(self, name, value):
        self.add({name: value})

    def write(self, fps=False):
        if fps:
            self.scalar('fps', self._compute_fps())
        if not self._metrics:
            return
        for output in self._outputs:
            output(self._metrics)
        self._metrics.clear()

    def _compute_fps(self):
        step = int(self._step) * self._multiplier
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration


class TerminalOutput:
    def __call__(self, summaries):
        # TODO aggregate
        # aggregate values in the same step
        scalar_summaries = collections.defaultdict(lambda: collections.defaultdict(list))
        for step, name, value in summaries:
            name = name.replace('/', '_')
            if len(value.shape) == 0:
                scalar_summaries[step][name].append(value.item())
        for step in scalar_summaries:
            scalars = {k: float(np.mean(v)) for k, v in scalar_summaries[step].items()}
            formatted = {k: self._format_value(v) for k, v in scalars.items()}
            print(f'[{step}]', ' / '.join(f'{k} {v}' for k, v in formatted.items()))
        # step = max(s for s, _, _, in summaries)
        # scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
        # formatted = {k: self._format_value(v) for k, v in scalars.items()}
        # print(f"[{step}]", " / ".join(f"{k} {v}" for k, v in formatted.items()))

    def _format_value(self, value):
        if value == 0:
            return '0'
        elif 0.01 < abs(value) < 10000:
            value = f'{value:.2f}'
            value = value.rstrip('0')
            # value = value.rstrip("0")
            value = value.rstrip('.')
            return value
        else:
            value = f'{value:.1e}'
            value = value.replace('.0e', 'e')
            value = value.replace('+0', '')
            value = value.replace('+', '')
            value = value.replace('-0', '-')
        return value


class JSONLOutput:
    def __init__(self, logdir):
        self._logdir = Path(logdir).expanduser()

    def __call__(self, summaries):
        # aggregate values in the same step
        scalar_summaries = collections.defaultdict(lambda: collections.defaultdict(list))
        for step, name, value in summaries:
            # name = name.replace('/', '_')
            if len(value.shape) == 0:
                scalar_summaries[step][name].append(value.item())
        for step in scalar_summaries:
            scalars = {k: np.mean(v) for k, v in scalar_summaries[step].items()}
            with (self._logdir / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': step, **scalars}) + '\n')


class WandbOutput:
    def __init__(self, fps=20, **kwargs):
        self._fps = fps
        wandb.init(**kwargs)

    def __call__(self, summaries):
        '''
            Dataformats:
            - scalar: float
            - image: numpy.ndarray with shape (C, H, W) or (H, W)
            - video: numpy.ndarray with shape (T, C, H, W)
        '''
        for step, name, value in summaries:
            value_dimension = len(value.shape)
            if value_dimension == 0:
                wandb.log({name: value}, step=step)
            elif value_dimension == 2 or value_dimension == 3:
                value = value.transpose((1, 2, 0)) if value_dimension==3 else value
                wandb.log({f'image/{name}': wandb.Image(value)}, step=step) # (H, W, C)
            elif value_dimension == 4:
                name = name if isinstance(name, str) else name.decode('utf-8')
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                # value = value.transpose((0, 3, 1, 2))
                wandb.log({f'video/{name}': wandb.Video(value, fps=self._fps, format='mp4')}, step=step) # (T, C, H, W)
