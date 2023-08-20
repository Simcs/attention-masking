import json
import pickle
import torch
import json
import random
from tqdm import tqdm
import argparse

from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.utils.visualization import save_roc_curve, save_precision_recall_curve

random.seed(4321)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--ucf_fold', type=int)
args = parser.parse_args()
ucf_fold = args.ucf_fold

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

print('loading train statistics...')
stat_fn = f'./data/ucf101/statistics/ucf101_{ucf_fold}_train_statistics_total.pkl'
with open(stat_fn, 'rb') as f:
    train_stat_total = pickle.load(f)
print('stat:', stat_fn)

# prepare statistics
means, precision = train_stat_total['mean'], train_stat_total['total_precision']
for ucf101_class in means:
    means[ucf101_class] = torch.Tensor(means[ucf101_class]).to(device)
precision = torch.Tensor(precision).to(device).float()

# ucv101 vs. hmdb51
# compute ucf101 mahalanobis distances
print('loading ucf101 embeddings...')
ucf_embedding_fn = f'./data/ucf101/embeddings/original/ucf101_{ucf_fold}_test_embeddings.jsonl'
print('ucf embedding:', ucf_embedding_fn)
ucf101_embs = []
with open(ucf_embedding_fn, 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        pre_logit = torch.Tensor(emb_js['penultimate'])
        ucf101_embs.append(pre_logit.view(1, -1))
ucf101_embs = torch.cat(ucf101_embs, dim=0)

print('computing ucf101 mahalanobis distances...')
ucf101_features = ucf101_embs.to(device)
ucf101_gaussian_scores = []
for mean in tqdm(means.values()):
    zero_f = ucf101_features - mean
    gau_term = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
    ucf101_gaussian_scores.append(gau_term.cpu().view(-1, 1))
ucf101_gaussian_scores = torch.cat(ucf101_gaussian_scores, dim=1)
ucf101_mahalanobis_distances, _ = ucf101_gaussian_scores.min(dim=1)
ucf101_mahalanobis_distances = ucf101_mahalanobis_distances.numpy().tolist()
    
print('computing gaussian scores finished.')
print('# of ucf101 embeddings:', len(ucf101_mahalanobis_distances))
    
# compute hmdb51 mahalanobis distances
print('loading mit embeddings...')
mit_embedding_fn = f'./data/mit/embeddings/original/mit_ucf101_{ucf_fold}_val_embeddings.jsonl'
print('hmdb embedding:', mit_embedding_fn)
mit_embs = []
with open(mit_embedding_fn, 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        pre_logit = torch.Tensor(emb_js['penultimate']).to(device)
        mit_embs.append(pre_logit.view(1, -1))
mit_embs = torch.cat(mit_embs, dim=0)

print('computing hmdb51 mahalanobis distances...')
# mit_features = mit_embs.to(device)
mit_mahalanobis_distances = []
for mit_features in mit_embs.chunk(2, dim=0):
    mit_gaussian_scores = []
    mit_features = mit_features.to(device)
    for mean in tqdm(means.values()):
        zero_f = mit_features - mean
        gau_term = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
        mit_gaussian_scores.append(gau_term.cpu().view(-1, 1))
    mit_gaussian_scores = torch.cat(mit_gaussian_scores, dim=1)
    mit_maha, _ = mit_gaussian_scores.min(dim=1)
    mit_mahalanobis_distances.extend(mit_maha.numpy())

print('computing gaussian scores finished.')
print('# of mit embeddings:', len(mit_mahalanobis_distances))

# min_len = min(len(ucf101_mahalanobis_distances), len(hmdb51_mahalanobis_distances))
# ucf101_mahalanobis_distances = random.sample(ucf101_mahalanobis_distances, min_len)
# hmdb51_mahalanobis_distances = random.sample(hmdb51_mahalanobis_distances, min_len)

print('computing ucf101 vs. mit ood scores...')
test_y = [0 for _ in range(len(ucf101_mahalanobis_distances))] + [1 for _ in range(len(mit_mahalanobis_distances))]
ood_scores = ucf101_mahalanobis_distances + mit_mahalanobis_distances
print('# of ucf101 and mit embeddings:', len(set(ood_scores)))

measure_performances(ucf101_mahalanobis_distances, mit_mahalanobis_distances)

fpr, tpr, ucf_hmdb_auroc_score = auroc(test_y, ood_scores)
pr, re, ucf_hmdb_aupr_score = aupr(test_y, ood_scores)
ucf_hmdb_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
print('auroc:', ucf_hmdb_auroc_score, 'aupr:', ucf_hmdb_aupr_score, 'fpr95:', ucf_hmdb_fpr95)
with open('./result/ood_scores/video/original/ucf101_vs_mit_maha.jsonl', 'w') as f:
    f.write(json.dumps({
        'auroc': ucf_hmdb_auroc_score,
        'aupr': ucf_hmdb_aupr_score,
        'fpr95': ucf_hmdb_fpr95,
    }))
save_roc_curve(fpr, tpr, './result/images/ucf101_vs_mit/maha_auroc.png')
save_precision_recall_curve(pr, re, './result/images/ucf101_vs_mit/maha_aupr.png')