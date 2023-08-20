import csv
import os
import time
import urllib
from functools import partial
from multiprocessing import Pool
from os import path
from typing import Any, Callable, Dict, Optional, Tuple, List
from pathlib import Path

from torch import Tensor

from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset


class MiT(VisionDataset):
    """Own implementation of Moments in Time v2 dataset bassed on torchvision.VisionDataset
    """
    
    def __init__(
        self,
        root: str,
        annotation_path: str,
        frames_per_clip: int,
        split: str = "train",
        frame_rate: Optional[int] = None,
        step_between_clips: int = 1,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ("avi", "mp4"),
        num_workers: int = 1,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        _audio_channels: int = 0,
        output_format: str = "TCHW",
    ) -> None:

        self.extensions = extensions

        self.root = Path(root)
        name = 'training' if split == 'train' else 'validation'
        self.split_folder = self.root / name
        self.split = verify_str_arg(split, arg="split", valid_values=["train", "val"])

        super().__init__(self.root)
        
        category_path = self.root / 'moments_categories.txt'
        # self.classes, class_to_idx = find_classes(self.split_folder)
        self.classes, class_to_idx = self._find_classes_from_category(category_path)
        self.samples = make_dataset(self.split_folder, class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
            output_format=output_format,
        )
        # self.full_video_clips = video_clips
        # self.indices = self._select_split(video_list, split)
        # self.video_clips = video_clips.subset(self.indices)        
        self.transform = transform

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.video_clips.metadata
    
    def _select_split(self, video_list: List[str], train: bool) -> List[int]:
        name = "training" if train else "validation"
        name = f"{name}Set.csv"
        f = os.path.join(self.root, name)
        selected_files = set()
        with open(f) as fid:
            data = fid.readlines()
            data = [x.strip().split(" ")[0] for x in data]
            data = [os.path.join(self.root, *x.split("/")) for x in data]
            selected_files.update(data)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices
    
    def _find_classes_from_category(self, category_path: Path) -> Tuple[List[str], Dict[str, int]]:
        classes, class_to_idx = [], {}
        with open(category_path, 'r') as f:
            for category in f.readlines():
                cls_name, idx = category.strip().split(',')
                classes.append(cls_name)
                class_to_idx[cls_name] = int(idx)
        
        return classes, class_to_idx

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    # return only video for better collate.
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label