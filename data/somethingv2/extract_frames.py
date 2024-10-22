"""
Credits to https://github.com/zhoubolei/TRN-pytorch/blob/master/extract_frames.py
"""

import os
import threading

NUM_THREADS = 100
VIDEO_ROOT = 'dataset/Something-Something/20bn-something-something-v2'         # Downloaded webm videos
IMAGE_SIZE = 240 #64
FRAME_ROOT = f'dataset/Something-Something/20bn-something-something-v2-frames-{IMAGE_SIZE}'  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, tmpl='%06d.jpg'):
    os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf scale={IMAGE_SIZE}:{IMAGE_SIZE} '
              f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')


def target(video_list):
    for video in video_list:
        targt_path = os.path.join(FRAME_ROOT, video[:-5])
        if not os.path.exists(targt_path):
            os.makedirs(targt_path)
        extract(video)


if not os.path.exists(VIDEO_ROOT):
    raise ValueError('Please download videos and set VIDEO_ROOT variable.')
if not os.path.exists(FRAME_ROOT):
    os.makedirs(FRAME_ROOT)

video_list = os.listdir(VIDEO_ROOT)
splits = list(split(video_list, NUM_THREADS))

threads = []
for i, split in enumerate(splits):
    thread = threading.Thread(target=target, args=(split,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
