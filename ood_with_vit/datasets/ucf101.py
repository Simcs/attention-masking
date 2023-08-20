from torchvision.datasets import UCF101

from typing import Tuple

from torch import Tensor


# audio-free version of kinetics for easy collation
class VideoOnlyUCF101(UCF101):

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]
        
        if self.transform is not None:
            video = self.transform(video)

        return video, class_index
    
class IndexUCF101(UCF101):

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]
        
        if self.transform is not None:
            video = self.transform(video)

        return video, class_index, idx