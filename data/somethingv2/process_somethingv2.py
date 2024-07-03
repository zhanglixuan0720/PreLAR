import json
import sys
import os
from tqdm import trange
from pathlib import Path

dataset_path = 'data/somethingv2/20bn-something-something-v2-frames-64'
lebels_base_ori = 'dataset/Something-Something/labels'
with open(Path(lebels_base_ori) / Path('labels.json'), 'r') as f:
    label = json.load(f)
with open(Path(lebels_base_ori) / Path('train.json'), 'r') as f:
    train = json.load(f)
with open(Path(lebels_base_ori) / Path('validation.json'), 'r') as f:
    val = json.load(f)
with open(Path(lebels_base_ori) / Path('test.json'), 'r') as f:
    test = json.load(f)

cnt = 0

category = [k for k in label.keys()]

train_txt = []
for i in trange(len(train), desc='train'):
    cur_index = train[i]['id']
    cur_label = train[i]['template']
    cur_label = cur_label.replace(']', '').replace('[', '')
    cur_id = label[cur_label]
    num_frames = len(os.listdir(os.path.join(dataset_path, cur_index)))
    if num_frames == 0:
        cnt += 1
    train_txt.append('%s %d %s' % (cur_index, num_frames, cur_id))

val_txt = []
for i in trange(len(val), desc='val'):
    cur_index = val[i]['id']
    cur_label = val[i]['template']
    cur_label = cur_label.replace(']', '').replace('[', '')
    cur_id = label[cur_label]
    num_frames = len(os.listdir(os.path.join(dataset_path, cur_index)))
    if num_frames == 0:
        cnt += 1
    val_txt.append('%s %d %s' % (cur_index, num_frames, cur_id))

# test_txt = []
# for i in trange(len(test), desc='test'):
#     cur_index = test[i]['id']
#     cur_label = test[i]['template']
#     cur_label = cur_label.replace(']', '').replace('[', '')
#     cur_id = label[cur_label]
#     num_frames = len(os.listdir(os.path.join(dataset_path, cur_index)))
#     if num_frames == 0:
#         cnt += 1
#     test_txt.append('%s %d %s' % (cur_index, num_frames, cur_id))

with open('train_video_folder.txt', 'w') as f:
    f.write('\n'.join(train_txt))
with open('val_video_folder.txt', 'w') as f:
    f.write('\n'.join(val_txt))
with open('train_val_video_folder.txt', 'w') as f:
    f.write('\n'.join(train_txt + val_txt))
# with open('test_folder.txt', 'w') as f:
#     f.write('\n'.join(test_txt))

with open('category.txt', 'w') as f:
    f.write('\n'.join(category))

print(cnt, 'empty')
