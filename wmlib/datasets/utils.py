from pathlib import Path
from .video import SomethingV2, DummyReplay, SomethingV2Flow
from ..core import ReplayWithoutAction
from ..core import Replay

def make_action_free_dataset(dataset_type,root_paths:dict=dict(),video_index_files:dict=dict(),segment_len=50,manual_labels=False,**kwargs):
    video_index_files = {dataset_type:video_index_files} if isinstance(video_index_files,str) else video_index_files
    root_paths = {dataset_type:root_paths} if isinstance(root_paths,str) else root_paths
    if dataset_type == 'replay':
        train_replay = ReplayWithoutAction(root_paths['replay'],**kwargs) # rquire load_directory, seed
    elif dataset_type == 'something':
        somethingv2_dataset = SomethingV2(
            root_path=root_paths['something'],
            list_file=f'data/somethingv2/{video_index_files["something"]}.txt',
            segment_len=segment_len,
            manual_labels=manual_labels,
        )
        train_replay = DummyReplay(somethingv2_dataset)
    elif dataset_type == 'something_flow':
        somethingv2_dataset = SomethingV2Flow(
            root_path=root_paths['something'],
            list_file=f'data/somethingv2/{video_index_files["something"]}.txt',
            segment_len=segment_len,
            manual_labels=manual_labels,
        )
        train_replay = DummyReplay(somethingv2_dataset)
    elif dataset_type == 'rlbench':
        train_replay = Replay(Path(root_paths['rlbench']), **kwargs)
    else:
        raise NotImplementedError
    return train_replay