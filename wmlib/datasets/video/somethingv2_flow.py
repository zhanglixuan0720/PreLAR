from torch.utils.data import Dataset

from PIL import Image
import os
import os.path
import numpy as np

from .utils import VideoRecord
from pathlib import Path

# NOTE: you can manually select videos with specific labels for training here, by default we use all videos
maunally_selected_labels = {
    "93": "Pushing something from left to right",
    "94": "Pushing something from right to left",
}


class SomethingV2Flow(Dataset):
    def __init__(self, root_path, list_file, segment_len=50, image_tmpl='{:06d}.jpg', manual_labels=False,flow_root=None,flow_tmpl='{:06d}.png'):
        self.root_path = Path(root_path).expanduser()
        self.list_file = list_file
        self.segment_len = segment_len
        self.image_tmpl = image_tmpl
        self.flow_tmpl = flow_tmpl
        if flow_root is None:
            self.flow_root = self.root_path / '../20bn-something-something-v2-frames-64-flow-rgb'
        else:
            self.flow_root = Path(flow_root).expanduser()
        self._parse_list(self.segment_len, maunally_selected_labels if manual_labels else None)

    def _parse_list(self, minlen, selected_labels=None):
        # check the frame number is large >segment_len:
        # usually it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= minlen and (
            (selected_labels is None) or (item[2] in selected_labels.keys()))]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    @property
    def total_steps(self):
        return sum([record.num_frames for record in self.video_list])

    def _load_image(self, directory, idx):
        # TODO: cache
        # image = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')
        image = Image.open(self.root_path / directory / self.image_tmpl.format(idx)).convert('RGB')
        return np.array(image)
    
    def _load_flow(self, directory, idx):
        idx = idx + 1 if idx == 1 else idx # no 1st frame flow
        image = Image.open(self.flow_root / directory / self.flow_tmpl.format(idx)).convert('RGB')
        return np.array(image)

    def _sample_index(self, record):
        return np.random.randint(0, record.num_frames - self.segment_len + 1) +1 # start from 1

    def get(self, record, ind):
        images = []
        p = ind
        for i in range(self.segment_len):
            seg_imgs = self._load_image(record.path, p)
            seg_flows = self._load_flow(record.path, p)
            images.append(np.concatenate([seg_imgs,seg_flows],axis=-1))
            if p < record.num_frames:
                p += 1
        # images = self.transform(images)
        return np.array(images)

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not (self.root_path / record.path / self.image_tmpl.format(1)).exists():
            print(self.root_path / record.path / self.image_tmpl.format(1))
        # while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
            # print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        segment_index = self._sample_index(record)
        segment = self.get(record, segment_index)
        return segment

    def __len__(self):
        return len(self.video_list)
