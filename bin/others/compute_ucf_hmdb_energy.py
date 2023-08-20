import json
import pickle
import torch
from tqdm import tqdm
import random
import argparse

from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.utils.visualization import save_roc_curve, save_precision_recall_curve

random.seed(4321)

parser = argparse.ArgumentParser()
parser.add_argument('--ucf_fold', type=int)
parser.add_argument('--hmdb_fold', type=int)
parser.add_argument('--T', default=1., type=float)
args = parser.parse_args()
ucf_fold, hmdb_fold, T = args.ucf_fold, args.hmdb_fold, args.T
print('T:', T)

def measure_performances(id_maha, ood_maha):
    results = [[] for _ in range(3)]
    for rn in tqdm(range(100)):
        random.seed(rn)
        
        min_len = min(len(id_maha), len(ood_maha))
        sampled_id_maha = random.sample(id_maha, min_len)
        sampled_ood_maha = random.sample(ood_maha, min_len)

        test_y = [0 for _ in range(len(sampled_id_maha))] + [1 for _ in range(len(sampled_ood_maha))]
        ood_scores = sampled_id_maha + sampled_ood_maha
        # max_ood_score = max(ood_scores)
        # ood_scores = [i / max_ood_score for i in ood_scores]
        # print('# of k400 and k600 embeddings:', len(set(ood_scores)))

        fpr, tpr, auroc_score = auroc(test_y, ood_scores)
        pr, re, aupr_score = aupr(test_y, ood_scores)
        fpr95 = fpr_at_95_tpr(test_y, ood_scores)
        results[0].append(auroc_score)
        results[1].append(aupr_score)
        results[2].append(fpr95)
    
    import numpy as np
    aurocs, auprs, fpr95s = np.array(results[0]), np.array(results[1]), np.array(results[2])
    print('        max  min  mean  std')
    print(f'auroc: {aurocs.max():.6f}  {aurocs.min():.6f}  {aurocs.mean():.6f} {aurocs.std():.6f}')
    print(f'aupr: {auprs.max():.6f}  {auprs.min():.6f}  {auprs.mean():.6f} {auprs.std():.6f}')
    print(f'fpr95: {fpr95s.max():.6f}  {fpr95s.min():.6f}  {fpr95s.mean():.6f} {fpr95s.std():.6f}')
    print('auroc argmax', aurocs.argmax(), 'argmin', aurocs.argmin())
    print('auprs argmax', auprs.argmax(), 'argmin', auprs.argmin())
    print('fpr95s argmax', fpr95s.argmax(), 'argmin', fpr95s.argmin())
    
# ucf101 vs. hmdb51
# compute ucf101 msp
print('loading ucf101 embeddings...')
ucf101_logits = []
with open(f'./data/ucf101/embeddings/original/ucf101_{ucf_fold}_test_embeddings.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        logit = torch.Tensor(emb_js['logit'])
        ucf101_logits.append(logit.view(1, -1))
ucf101_logits = torch.cat(ucf101_logits, dim=0)
ucf101_energy = T * torch.logsumexp(ucf101_logits / T, dim=1)
ucf101_energy = (-ucf101_energy.numpy()).tolist()
    
# compute hmdb51 msp
print('loading hmdb51 embeddings...')
hmdb51_logits = []
with open(f'./data/hmdb51/embeddings/original/hmdb51_{hmdb_fold}_ucf101_{ucf_fold}_train_embeddings.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        logit = torch.Tensor(emb_js['logit'])
        hmdb51_logits.append(logit.view(1, -1))
hmdb51_logits = torch.cat(hmdb51_logits, dim=0)
hmdb51_energy = T * torch.logsumexp(hmdb51_logits / T, dim=1)
hmdb51_energy = (-hmdb51_energy.numpy()).tolist()

measure_performances(ucf101_energy, hmdb51_energy)

print('computing ucf101 vs. hmdb51 ood scores...')
test_y = [0 for _ in range(len(ucf101_energy))] + [1 for _ in range(len(hmdb51_energy))]
ood_scores = ucf101_energy + hmdb51_energy

import matplotlib.pyplot as plt

plt.hist((ucf101_energy, hmdb51_energy), histtype='bar')
plt.title('ucf101 + hmdb51 energy')
plt.savefig('test.png')

fpr, tpr, ucf_hmdb_auroc_score = auroc(test_y, ood_scores)
pr, re, ucf_hmdb_aupr_score = aupr(test_y, ood_scores)
ucf_hmdb_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
print('auroc:', ucf_hmdb_auroc_score, 'aupr:', ucf_hmdb_aupr_score, 'fpr95:', ucf_hmdb_fpr95)
with open('./result/ood_scores/video/original/ucf101_vs_hmdb51_msp.jsonl', 'w') as f:
    f.write(json.dumps({
        'auroc': ucf_hmdb_auroc_score,
        'aupr': ucf_hmdb_aupr_score,
        'fpr95': ucf_hmdb_fpr95,
    }))
save_roc_curve(fpr, tpr, './result/images/ucf101_vs_hmdb51/msp_auroc.png')
save_precision_recall_curve(pr, re, './result/images/ucf101_vs_hmdb51/msp_aupr.png')