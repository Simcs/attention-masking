from torchvision.datasets import Kinetics

from typing import Tuple, Union, List

from torch import Tensor

import csv
import os
import time
import urllib
import warnings
from functools import partial
from multiprocessing import Pool
from os import path
from typing import Any, Callable, Dict, Optional, Tuple, cast

from torch import Tensor

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.vision import VisionDataset


# # audio-free version of kinetics for easy collation
class VideoOnlyKinetics(Kinetics):

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        # comment out on torchvision 0.13.1
        # if not self._legacy:
        #     # [T,H,W,C] --> [T,C,H,W]
        #     video = video.permute(0, 3, 1, 2)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label