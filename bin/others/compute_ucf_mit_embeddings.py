import os
import time
from pprint import pprint
from pathlib import Path
from venv import create
from tqdm import tqdm
import pickle
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import video_transformer.data_transform as T
from video_transformer.transformer import ClassificationHead

from video_transformer.video_transformer import ViViT
from ood_with_vit.datasets.kinetics import VideoOnlyKinetics
from ood_with_vit.datasets.ucf101 import VideoOnlyUCF101
from ood_with_vit.datasets.hmdb51 import VideoOnlyHMDB51
from ood_with_vit.datasets.mit import MiT


def replace_state_dict(state_dict):
    for old_key in list(state_dict.keys()):
        if old_key.startswith('model'):
            new_key = old_key[6:]
            state_dict[new_key] = state_dict.pop(old_key)
        else:
            new_key = old_key[9:]
            state_dict[new_key] = state_dict.pop(old_key)

def init_from_pretrain_(module, pretrained, init_module):
    if torch.cuda.is_available():
        state_dict = torch.load(pretrained)
    else:
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
  
    if init_module == 'transformer':
        replace_state_dict(state_dict)
    elif init_module == 'cls_head':
        replace_state_dict(state_dict)
    else:
        raise TypeError(f'pretrained weights do not include the {init_module} module')
    msg = module.load_state_dict(state_dict, strict=False)
    return msg


def create_model():
    num_frames = 8
    frame_interval = 32
    # num_class = 400
    num_class = 101
    arch = 'vivit' # turn to vivit for initializing vivit model

    # pretrain_pth = './logs/vivit/vivit_model.pth'
    
    # # fold 1
    # pretrain_pth = './logs/results/' + \
    #     'vivit_pretrained_ucf101_fp32_fold_1_aug/ckpt/' + \
    #         '2023-03-13 23:03:32_ep_1_top1_acc_0.915.pth'
    # # fold 2
    pretrain_pth = './logs/results/' + \
        'vivit_pretrained_ucf101_fp32_fold_2_aug/ckpt/' + \
            '2023-03-15 10:15:54_ep_14_top1_acc_0.931.pth'
    # # fold 3
    pretrain_pth = './logs/results/' + \
        'vivit_pretrained_ucf101_fp32_fold_3_aug/ckpt/' + \
            '2023-03-15 03:25:46_ep_13_top1_acc_0.894.pth'
            
    num_frames = num_frames * 2
    frame_interval = frame_interval // 2
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

    cls_head = ClassificationHead(num_classes=num_class, in_channels=768)
    # msg_trans = init_from_pretrain_(model, pretrain_pth, init_module='transformer')
    msg_cls = init_from_pretrain_(cls_head, pretrain_pth, init_module='cls_head')
    print(f'load model finished, the missing key of cls is:{msg_cls[0]}')
    
    return model, cls_head


def compute_embeddings(args):
    dataset, fold, split, force = args.dataset, args.fold, args.split, args.force
    print(f'computing {dataset} fold {fold} split {split}...')
    
    
    embeddings_filename = f'./data/{dataset}/embeddings/{dataset}_{split}_embeddings.jsonl'
    # embeddings_filename = os.path.expanduser(embeddings_filename)
    
    if not force and os.path.exists(embeddings_filename):
        return
    
    # prepare metadata
    if dataset == 'ucf101':
        with open(f'./data/{dataset}/metadata/{dataset}_{fold}_{split}_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    else:
        print(f'./data/{dataset}/metadata/{dataset}_{split}_metadata.pkl')
        with open(f'./data/{dataset}/metadata/{dataset}_{split}_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
    with open(f'./data/ucf101/metadata/ucf101_1_test_metadata.pkl', 'rb') as f:
        ucf_metadata = pickle.load(f)
        
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.create_video_transform(
        input_size=224,
        is_training=False,
        interpolation='bicubic',
        mean=mean,
        std=std,
    )

    # prepare datasets and dataloaders
    if dataset == 'ucf101':
        dataset_root = f'~/workspace/dataset/ucf101'
        dataset_root = os.path.expanduser(dataset_root)
        annotation_path = '~/workspace/dataset/ucfTrainTestlist'
        annotation_path = os.path.expanduser(annotation_path)
  
        if split == 'train':
            target_ds = VideoOnlyUCF101(
                root=dataset_root,
                annotation_path=annotation_path,
                frames_per_clip=16,
                step_between_clips=1,
                frame_rate=2,
                fold=fold,
                train=True,
                transform=transform,
                num_workers=8,
                output_format='TCHW',
                _precomputed_metadata=metadata,
            )
        else:
            target_ds = VideoOnlyUCF101(
                root=dataset_root,
                annotation_path=annotation_path,
                frames_per_clip=16,
                step_between_clips=1,
                frame_rate=2,
                fold=fold,
                train=False,
                transform=transform,
                num_workers=8,
                output_format='TCHW',
                _precomputed_metadata=metadata,
            )
    
    elif dataset == 'mit':
        dataset_root = f'~/workspace/dataset/Moments_in_Time_Raw'
        dataset_root = os.path.expanduser(dataset_root)
        annotation_path = '~/workspace/datset/Moments_in_Time_Raw/validationSet.csv'
        annotation_path = os.path.expanduser(annotation_path)
  
        if split == 'train':
            target_ds = MiT(
                root=dataset_root,
                annotation_path=annotation_path,
                frames_per_clip=16,
                step_between_clips=1,
                frame_rate=6,
                split='train',
                transform=transform,
                num_workers=8,
                output_format='TCHW',
                _precomputed_metadata=metadata,
            )
        else:
            target_ds = MiT(
                root=dataset_root,
                annotation_path=annotation_path,        
                frames_per_clip=16,
                step_between_clips=1,
                frame_rate=6,
                split='val',
                transform=transform,
                num_workers=8,
                output_format='TCHW',
                _precomputed_metadata=metadata,
            )
 
    ucf101_root = f'~/workspace/dataset/ucf101'
    ucf101_root = os.path.expanduser(ucf101_root)
    ucf101_annotation_path = '~/workspace/dataset/ucfTrainTestlist'
    ucf101_annotation_path = os.path.expanduser(ucf101_annotation_path)
       
    ucf101_ds = VideoOnlyUCF101(
        root=ucf101_root,
        annotation_path=ucf101_annotation_path,
        frames_per_clip=16,
        step_between_clips=1,
        frame_rate=2,
        fold=fold,
        transform=transform,
        num_workers=8,
        _precomputed_metadata=ucf_metadata,
    )
    # print('target_ds:', target_ds.classes)
    # print('ucf101_ds:', ucf101_ds.classes)
    # input()
    
    dl = DataLoader(
        dataset=target_ds,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )
    
    # prepare models
    model, cls_head = create_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cls_head = model.to(device), cls_head.to(device)
    
    # compute embeddings
    n_correct, n_total = 0, 0
    id, cache_rate = 0, 100
    logs = []

    model.eval()
    cls_head.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dl)):
            if batch_idx % cache_rate == 0:
                with open(embeddings_filename, 'a+') as f:
                    for log in logs:
                        f.write(json.dumps(log) + '\n')
                logs = []
                
            x, y = x.to(device), y.to(device)
            pre_logits = model(x)
            logits = cls_head(pre_logits)
            _, predicted = logits.max(1)

            for pre_logit, logit, gt, pred in zip(pre_logits, logits, y, predicted):
                pre_logit = pre_logit.detach().cpu().numpy().tolist()
                logit = logit.detach().cpu().numpy().tolist()
                gt_label = target_ds.classes[gt.item()]
                pred_label = ucf101_ds.classes[pred.item()]
                # print('gt:', gt_label, 'pred:', pred_label, gt_label == pred_label)
                logs.append({
                    'id': id, 
                    'gt': gt_label, 
                    'pred': pred_label, 
                    'penultimate': pre_logit,
                    'logit': logit,
                })
                id += 1

            n_total += y.size(0)
            n_correct += predicted.eq(y).sum().item()

        with open(embeddings_filename, 'a+') as f:
            for log in logs:
                f.write(json.dumps(log) + '\n')
        logs = []
        
        test_accuracy = 100. * n_correct / n_total
        print(f'Test Acc: {test_accuracy:.3f}% ({n_correct}/{n_total})')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ucf101', 'mit'])
    parser.add_argument('--fold', type=int)
    parser.add_argument('--split', choices=['train', 'val', 'test'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    force = args.force
    compute_embeddings(args)