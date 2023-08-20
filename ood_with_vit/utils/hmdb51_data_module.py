import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from ood_with_vit.datasets.hmdb51 import VideoOnlyHMDB51

import video_transformer.data_transform as T

class Collator(object):

	def __init__(self, objective):
		self.objective = objective
	
	def collate(self, minibatch):
		image_list = []
		label_list = []
		mask_list = []
		marker_list = []
		for record in minibatch:
			image_list.append(record[0])
			label_list.append(record[1])
			if self.objective == 'mim':
				mask_list.append(record[2])
				marker_list.append(record[3])
		minibatch = []
		minibatch.append(torch.stack(image_list))
		if self.objective == 'mim':
			minibatch.append(torch.stack(label_list))
			minibatch.append(torch.stack(mask_list))
			minibatch.append(marker_list)
		else:
			label = np.stack(label_list)
			minibatch.append(torch.from_numpy(label))
		
		return minibatch

class HMDB51DataModule(pl.LightningDataModule):
	def __init__(self, 
				 configs,
				 train_ann_path,
				 val_ann_path=None,
				 test_ann_path=None,
				 ):
		super().__init__()
		self.train_ann_path = train_ann_path
		self.val_ann_path = val_ann_path
		self.test_ann_path = test_ann_path
		self.configs = configs

	def get_dataset(self, annotation_path, transform, train, temporal_sample=None):
		hmdb51_root = '~/workspace/dataset/hmdb51'
		hmdb51_root = os.path.expanduser(hmdb51_root)
		annotation_path = '~/workspace/dataset/testTrainMulti_7030_splits'
		annotation_path = os.path.expanduser(annotation_path)
  
		dataset = VideoOnlyHMDB51(
			root=hmdb51_root,
			annotation_path=annotation_path,
			frames_per_clip=self.configs.num_frames,
			step_between_clips=self.configs.frame_interval,
			frame_rate=2,
			train=train,
			transform=transform,
			num_workers=self.configs.num_workers,
			output_format='TCHW',
		)
		
		return dataset

	def setup(self, stage):
		color_jitter = 0.4
		scale = None
		# use imagenet mean, std
		mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
		
		train_transform = T.create_video_transform(
			input_size=self.configs.img_size,
			is_training=True,
			interpolation='bicubic',
			mean=mean,
			std=std,
			objective=self.configs.objective,
		)
  
		# train_transform = T.create_video_transform(
		# 	objective=self.configs.objective,
		# 	input_size=self.configs.img_size,
		# 	is_training=True,
		# 	scale=scale,
		# 	hflip=0.5,
		# 	color_jitter=color_jitter,
		# 	auto_augment=self.configs.auto_augment,
		# 	interpolation='bicubic',
		# 	mean=mean,
		# 	std=std
   		# )
  
		train_temporal_sample = T.TemporalRandomCrop(
			self.configs.num_frames * self.configs.frame_interval)
			
		self.train_dataset = self.get_dataset(
			annotation_path=self.train_ann_path,
			transform=train_transform,
			train=True,
			# train_temporal_sample
   		)
  
		if self.val_ann_path is not None:
			val_transform = T.create_video_transform(
				input_size=self.configs.img_size,
				is_training=False,
				interpolation='bicubic',
				mean=mean,
				std=std,
    		)
			val_temporal_sample = T.TemporalRandomCrop(
				self.configs.num_frames * self.configs.frame_interval)
   
			self.val_dataset = self.get_dataset(
				annotation_path=self.val_ann_path,
				transform=val_transform,
				train=False,
				# val_temporal_sample
    		)
		
		if self.test_ann_path is not None:
			# need to update
			test_transform = T.Compose([
				T.Resize(scale_range=(-1, 256)),
				T.ThreeCrop(size=self.configs.img_size),
				T.ToTensor(),
				T.Normalize(mean, std),
			])
			test_temporal_sample = T.TemporalRandomCrop(
				self.configs.num_frames * self.configs.frame_interval)
			self.test_dataset = self.get_dataset(
				annotation_path=self.test_ann_path,
				transform=test_transform,
				train=False,
				# test_temporal_sample
    		)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.configs.batch_size,
			num_workers=self.configs.num_workers,
			collate_fn=Collator(self.configs.objective).collate,
			shuffle=True,
			drop_last=True, 
			pin_memory=True
		)
	
	def val_dataloader(self):
		if self.val_ann_path is not None:
			return DataLoader(
				self.val_dataset,
				batch_size=self.configs.batch_size,
				num_workers=self.configs.num_workers,
				collate_fn=Collator(self.configs.objective).collate,
				shuffle=False,
				drop_last=False,
			)
	
	def test_dataloader(self):
		if self.test_ann_path is not None:
			return DataLoader(
				self.test_dataset,
				batch_size=self.configs.batch_size,
				num_workers=self.configs.num_workers,
				collate_fn=Collator(self.configs.objective).collate,
				shuffle=False,
				drop_last=False,
			)