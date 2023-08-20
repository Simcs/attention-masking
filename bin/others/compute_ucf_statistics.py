import json
import pickle
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

from torchvision.datasets import Kinetics
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance

from ood_with_vit.datasets.ucf101 import VideoOnlyUCF101

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int)
args = parser.parse_args()
fold = args.fold

correct_penultimates = {}
# wrong_penultimates = {}

filename = f'./data/ucf101/ucf101_{fold}_train_embeddings.jsonl'
total = 0
with open(filename, 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        penultimate_feature = torch.Tensor(emb_js['penultimate'])
        gt_label, pred_label = emb_js['gt'], emb_js['pred']
        if gt_label not in correct_penultimates:
            correct_penultimates[gt_label] = []
        # if gt_label not in wrong_penultimates:
        #     wrong_penultimates[gt_label] = []
            
        # if gt_label == pred_label:
        correct_penultimates[gt_label].append(penultimate_feature.view(1, -1))
        # else:
        #     wrong_penultimates[gt_label].append(penultimate_feature)
        total += 1

# ucf101_root = f'~/workspace/dataset/ucf101'
# ucf101_root = os.path.expanduser(ucf101_root)
# ucf101_annotation_path = '~/workspace/dataset/ucfTrainTestlist'
# ucf101_annotation_path = os.path.expanduser(ucf101_annotation_path)
# with open(f'./data/ucf101/ucf101_1_test_metadata.pkl', 'rb') as f:
#     ucf_metadata = pickle.load(f)
    
# ucf101_ds = VideoOnlyUCF101(
#     root=ucf101_root,
#     annotation_path=ucf101_annotation_path,
#     frames_per_clip=16,
#     step_between_clips=1,
#     frame_rate=2,
#     fold=2,
#     num_workers=8,
#     _precomputed_metadata=ucf_metadata,
# )
# print(ucf101_ds.classes)

# error_classes = []
# for c in ucf101_ds.classes:
#     if c not in correct_penultimates.keys():
#         error_classes.append(c)
# print(error_classes, len(error_classes))
# print(len(correct_penultimates.keys()), total)
# input()

correct_means = {c: None for c in correct_penultimates.keys()}
for c in correct_penultimates.keys():
    correct_penultimates[c] = torch.cat(correct_penultimates[c], dim=0)
    correct_means[c] = torch.mean(correct_penultimates[c], dim=0)


# total
group_lasso = EmpiricalCovariance(assume_centered=False)
X = []
for c in correct_penultimates.keys():
    X.append(correct_penultimates[c] - correct_means[c])
X = torch.cat(X, dim=0).numpy()
group_lasso.fit(X)
total_precision = torch.from_numpy(group_lasso.precision_).float()
print('covariance norm:', np.linalg.norm(group_lasso.precision_))
result_total = {
    'mean': correct_means,
    'total_precision': total_precision.numpy().tolist(),
}

# classwise
classwise_precisions = {}
for c in correct_penultimates.keys():
    group_lasso = EmpiricalCovariance(assume_centered=False)
    X = (correct_penultimates[c] - correct_means[c]).numpy()
    group_lasso.fit(X)
    precision = torch.from_numpy(group_lasso.precision_).float()
    if np.linalg.norm(precision) < 100:
        continue
    classwise_precisions[c] = precision
norms = [np.linalg.norm(classwise_precisions[c]) for c in classwise_precisions.keys()]
print('covairance norms:', norms, len(norms))
result_classwise = {}
for c in correct_means.keys():
    if c not in classwise_precisions:
        continue
    result_classwise[c] = {
        'mean': correct_means[c].numpy().tolist(),
        'classwise_precision': classwise_precisions[c].tolist(),
    }


with open(f'./data/ucf101/statistics/ucf101_{fold}_train_statistics_total.pkl', 'wb') as f:
    pickle.dump(result_total, f)

with open(f'./data/ucf101/statistics/ucf101_{fold}_train_statistics_classwise.pkl', 'wb') as f:
    pickle.dump(result_classwise, f)