from typing import Tuple

import timm
from ml_collections.config_dict import ConfigDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from . import BaseTrainer

from video_transformer.video_transformer import ViViT
from video_transformer.transformer import PatchEmbed, TransformerContainer, ClassificationHead


class ViViT_Finetune_UCF101_Trainer(BaseTrainer):
    
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        
    def _replace_state_dict(state_dict):
        for old_key in list(state_dict.keys()):
            if old_key.startswith('model'):
                new_key = old_key[6:]
                state_dict[new_key] = state_dict.pop(old_key)
            else:
                new_key = old_key[9:]
                state_dict[new_key] = state_dict.pop(old_key)
                
    def _init_from_pretrain_(module, pretrained, init_module):
        if torch.cuda.is_available():
            state_dict = torch.load(pretrained)
        else:
            state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
        if init_module == 'transformer':
            replace_state_dict(state_dict)
        elif init_module == 'cls_head':
            replace_state_dict(state_dict)
        else:
            raise TypeError(f'pretrained weights do not include the {init_module} module')
        msg = module.load_state_dict(state_dict, strict=False)
        return msg 
    
    def _create_model(self) -> nn.Module:
        num_frames = 16
        frame_interval = 16
        num_class = 400
        pretrain_pth = './logs/vivit/vivit_model.pth'
        
        model = ViViT(
            num_frames=num_frames,
            img_size=224,
            patch_size=16,
            embed_dims=768,
            in_channels=3,
            attention_type='fact_encoder',
            return_cls_token=True,
            pretrain_pth=pretrain_pth,
            weights_from='kinetics',
        )
        
        cls_head = ClassificationHead(
            num_classes=num_class,
            in_channels=768,
        )
        msg_cls = init_from_pretrain_(cls_head, pretrain_pth, init_module='cls_head')
        model = model.to(device)
        cls_head = cls_head.to(device)
        print(f'load model finished, the missing key of cls is:{msg_cls[0]}')
    
    def _create_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        dataset_mean, dataset_std = self.config.dataset.mean, self.config.dataset.std
        dataset_root = self.config.dataset.root
        img_size = self.config.model.img_size
        
        print('==> Preparing data..')        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        trainset = CIFAR10(
            root=dataset_root,
            train=True, 
            download=True, 
            transform=transform_train
        )
        trainloader = DataLoader(
            dataset=trainset, 
            batch_size=self.config.train.batch_size, 
            shuffle=True, 
            num_workers=8
        )
        
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        testset = CIFAR10(
            root=dataset_root, 
            train=False, 
            download=True, 
            transform=transform_test
        )
        testloader = DataLoader(
            dataset=testset, 
            batch_size=self.config.eval.batch_size, 
            shuffle=False, 
            num_workers=8
        )
        return trainloader, testloader