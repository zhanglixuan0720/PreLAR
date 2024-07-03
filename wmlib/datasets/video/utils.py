from torch.utils.data import Dataset, DataLoader, IterableDataset

import numpy as np
from ...core import seed_worker


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    def __str__(self):
        return str(self._data)


# a mixture of datasets
class Mixture(Dataset):

    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        self.weights = weights
        self.lengths = [len(d) for d in datasets]
        self.total_length = sum(self.lengths)

    def __getitem__(self, index):
        if self.weights is None:
            dataset_index = np.random.randint(len(self.datasets))
        else:
            dataset_index = np.random.choice(len(self.datasets), p=self.weights)
        return self.datasets[dataset_index][index % self.lengths[dataset_index]]

    def __len__(self):
        return self.total_length

    @property
    def total_steps(self):
        return sum([d.total_steps for d in self.datasets])


class DummyReplay:  # A wrapper make datasets behave like replay buffers

    def __init__(self, video_dataset) -> None:
        self.video_dataset = video_dataset

    def _generate_chunks(self, length):
        while True:
            ind = np.random.randint(len(self.video_dataset))
            image = self.video_dataset[ind]
            action = np.zeros((image.shape[0], 1), dtype=np.float32)
            is_first = np.zeros((image.shape[0]), dtype=bool)
            is_first[0] = True
            chunk = {
                'image': image,
                'action': action,
                'is_first': is_first,
            }
            # T,H,W,C -> T,C,H,W
            if len(chunk['image'].shape) == 4:
                chunk['image'] = chunk['image'].transpose(0, 3, 1, 2)
            yield chunk

    def dataset(self, batch, length, pin_memory=True, num_workers=8, **kwargs):
        generator = lambda: self._generate_chunks(length)

        class ReplayDataset(IterableDataset):
            def __iter__(self):
                return generator()

        dataset = ReplayDataset()
        dataset = DataLoader(
            dataset,
            batch,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
            num_workers=num_workers,
            **kwargs
        )
        return dataset

    @property
    def stats(self):
        return {
            "total_steps": 0,
            "total_episodes": 0,
            "loaded_steps": self.video_dataset.total_steps,
            "loaded_episodes": len(self.video_dataset),
        }
