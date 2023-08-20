import json
import pickle
import torch
from tqdm import tqdm
import random
import argparse

from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.utils.visualization import save_roc_curve, save_precision_recall_curve, save_histogram

random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--T', default=1., type=float)
args = parser.parse_args()
T = args.T
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
    
# kinetics400 vs. kinetics600
# compute k400 mahalanobis distances
print('loading k400 embeddings...')
k400_val = []
with open('./data/kinetics/embeddings/original/k400_val_embeddings.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        logit = torch.Tensor(emb_js['logit'])
        k400_val.append(logit.view(1, -1))
k400_logits = torch.cat(k400_val, dim=0)
k400_energy = T * torch.logsumexp(k400_logits / T, dim=1)
k400_energy = (-k400_energy.numpy()).tolist()
    
# compute k600 mahalanobis distances
print('loading k600 embeddings...')
k600_original = []
with open('./data/kinetics/embeddings/original/k600_val_embeddings_deduplicated.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        logit = torch.Tensor(emb_js['logit'])
        k600_original.append(logit.view(1, -1))
k600_logits = torch.cat(k600_original, dim=0)
k600_energy = T * torch.logsumexp(k600_logits / T, dim=1)
k600_energy = (-k600_energy.numpy()).tolist()

measure_performances(k400_energy, k600_energy)

print('computing k400 vs. k600 ood scores...')
test_y = [0 for _ in range(len(k400_energy))] + [1 for _ in range(len(k600_energy))]
ood_scores = k400_energy + k600_energy

id_label, ood_label, detector = 'K400', 'K600_excl', 'Energy'
save_histogram(
    id=k400_energy, 
    ood=k600_energy, 
    id_label=id_label, 
    ood_label=ood_label, 
    detector=detector, 
    path=f'histograms/{id_label}+{ood_label}_{detector}.png',
    auc=76.02
)

fpr, tpr, k600_auroc_score = auroc(test_y, ood_scores)
pr, re, k600_aupr_score = aupr(test_y, ood_scores)
k600_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
print('auroc:', k600_auroc_score, 'aupr:', k600_aupr_score, 'fpr95:', k600_fpr95)
with open('./result/ood_scores/video/original/k400_vs_k600_energy.jsonl', 'w') as f:
    f.write(json.dumps({
        'auroc': k600_auroc_score,
        'aupr': k600_aupr_score,
        'fpr95': k600_fpr95,
    }))
save_roc_curve(fpr, tpr, './result/images/k400_vs_k600/energy_auroc.png')
save_precision_recall_curve(pr, re, './result/images/k400_vs_k600/energy_aupr.png')

# compute k700 mahalanobis distances
print('computing k700 mahalanobis distances...')
k700_original = []
with open('./data/kinetics/embeddings/original/k700-2020_val_embeddings_deduplicated.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        logit = torch.Tensor(emb_js['logit'])
        k700_original.append(logit.view(1, -1))
k700_logits = torch.cat(k700_original, dim=0)
k700_energy = T * torch.logsumexp(k700_logits / T, dim=1)
k700_energy = (-k700_energy.numpy()).tolist()

measure_performances(k400_energy, k700_energy)

print('computing k400 vs. k700 ood scores...')
test_y = [0 for _ in range(len(k400_energy))] + [1 for _ in range(len(k700_energy))]
ood_scores = k400_energy + k700_energy

id_label, ood_label, detector = 'K400', 'K700_excl', 'Energy'
save_histogram(
    id=k400_energy, 
    ood=k600_energy, 
    id_label=id_label, 
    ood_label=ood_label, 
    detector=detector, 
    path=f'histograms/{id_label}+{ood_label}_{detector}.png',
    auc=78.14
)

fpr, tpr, k700_auroc_score = auroc(test_y, ood_scores)
pr, re, k700_aupr_score = aupr(test_y, ood_scores)
k700_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
print('k400 vs. k700:')
print('auroc:', k700_auroc_score, 'aupr:', k700_aupr_score, 'fpr95:', k700_fpr95)
with open('./result/ood_scores/video/original/k400_vs_k700_energy.jsonl', 'w') as f:
    f.write(json.dumps({
        'auroc': k700_auroc_score,
        'aupr': k700_aupr_score,
        'fpr95': k700_fpr95,
    }))
save_roc_curve(fpr, tpr, './result/images/k400_vs_k700/energy_auroc.png')
save_precision_recall_curve(pr, re, './result/images/k400_vs_k700/energy_aupr.png')