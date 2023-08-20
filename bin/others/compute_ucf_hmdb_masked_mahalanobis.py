import json
import pickle
import torch
from tqdm import tqdm
from pathlib import Path
import argparse
import random

from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.utils.visualization import save_roc_curve, save_precision_recall_curve


ucf_fold, hmdb_fold = 1, 1

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
    print('        max  min  mean median std')
    print(f'auroc: {aurocs.max():.6f}  {aurocs.min():.6f}  {aurocs.mean():.6f} {np.median(aurocs):.6f} {aurocs.std():.6f}')
    print(f'aupr: {auprs.max():.6f}  {auprs.min():.6f}  {auprs.mean():.6f} {np.median(auprs):.6f} {auprs.std():.6f}')
    print(f'fpr95: {fpr95s.max():.6f}  {fpr95s.min():.6f}  {fpr95s.mean():.6f} {np.median(fpr95s):.6f} {fpr95s.std():.6f}')
    print('auroc argmax', aurocs.argmax(), 'argmin', aurocs.argmin())
    print('auprs argmax', auprs.argmax(), 'argmin', auprs.argmin())
    print('fpr95s argmax', fpr95s.argmax(), 'argmin', fpr95s.argmin())

def get_graph_filename(
    dataset1: str,
    dataset2: str,
    args,
):
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    
    image_dir = Path('./result') / 'images' / 'masked' / f'{dataset1}_vs_{dataset2}'
    image_dir.mkdir(exist_ok=True)
    image_fn = image_dir / f'{head_fusion}_{discard_ratio}'
    if spatial_masking and not temporal_masking:
        if 'ratio' in spatial_mask_method:
            image_fn = image_fn.parent / f'spatial_{image_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}'
        else:
            image_fn = image_fn.parent / f'spatial_{image_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}'
    elif not spatial_masking and temporal_masking:
        if 'ratio' in temporal_mask_method:
            image_fn = image_fn.parent / f'temporal_{image_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}'
        else:
            image_fn = image_fn.parent / f'temporal_{image_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}'
    elif spatial_masking and temporal_masking:
        if 'ratio' in spatial_mask_method:
            if 'ratio' in temporal_mask_method:
                image_fn = image_fn.parent / (f'spatiotemporal_{image_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}')
            else:
                image_fn = image_fn.parent / (f'spatiotemporal_{image_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}')
        else:
            if 'ratio' in temporal_mask_method:
                image_fn = image_fn.parent / (f'spatiotemporal_{image_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}')
            else:
                image_fn = image_fn.parent / (f'spatiotemporal_{image_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}')
                
    return image_fn

def get_result_filename(
    dataset1: str,
    dataset2: str,
    args,
):
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    
    result_dir = Path('./result') / 'ood_scores' / 'video' / 'masked' / f'{dataset1}_vs_{dataset2}'
    result_dir.mkdir(exist_ok=True)
    result_fn = result_dir / f'{head_fusion}_{discard_ratio}'
    if spatial_masking and not temporal_masking:
        if 'ratio' in spatial_mask_method:
            result_fn = result_fn.parent / f'spatial_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}.jsonl'
        else:
            result_fn = result_fn.parent / f'spatial_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}.jsonl'
    elif not spatial_masking and temporal_masking:
        if 'ratio' in temporal_mask_method:
            result_fn = result_fn.parent / f'temporal_{result_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}.jsonl'
        else:
            result_fn = result_fn.parent / f'temporal_{result_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}.jsonl'
    elif spatial_masking and temporal_masking:
        if 'ratio' in spatial_mask_method:
            if 'ratio' in temporal_mask_method:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_'\
                    + f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_'\
                    + f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
        else:
            if 'ratio' in temporal_mask_method:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
                
    return result_fn

def get_embeddings_filename(
    dataset: str,
    fold: int,
    split: str,
    args,
):
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    
    if dataset == 'ucf101':
        embeddings_dir = Path('./data') / dataset / 'embeddings' / 'masked' / f'{dataset}_{fold}_{split}'
    else:
        # TODO: change ucf_fold
        # ucf_fold = 1
        embeddings_dir = Path('./data') / dataset / 'embeddings' / 'masked' / f'{dataset}_{fold}_ucf101_{ucf_fold}_{split}'
        
    emb_fn = embeddings_dir / f'{head_fusion}_{discard_ratio}'
    if spatial_masking and not temporal_masking:
        if 'ratio' in spatial_mask_method:
            emb_fn = emb_fn.parent / f'spatial_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}.jsonl'
        else:
            emb_fn = emb_fn.parent / f'spatial_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}.jsonl'
    elif not spatial_masking and temporal_masking:
        if 'ratio' in temporal_mask_method:
            emb_fn = emb_fn.parent / f'temporal_{emb_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}.jsonl'
        else:
            emb_fn = emb_fn.parent / f'temporal_{emb_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}.jsonl'
    elif spatial_masking and temporal_masking:
        if 'ratio' in spatial_mask_method:
            if 'ratio' in temporal_mask_method:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
        else:
            if 'ratio' in temporal_mask_method:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
                
    return emb_fn

def get_statistics_filename(
    dataset: str,
    fold: int,
    split: str,
    args,
):
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    
    statistics_dir = Path('./data') / dataset / 'statistics' / 'masked' / f'{dataset}_{fold}_train'
    # statistics_dir = Path('./data') / dataset / 'statistics' / 'masked' / f'{dataset}_{split}'
    stat_fn = statistics_dir / f'{head_fusion}_{discard_ratio}'
    if spatial_masking and not temporal_masking:
        if 'ratio' in spatial_mask_method:
            stat_fn = stat_fn.parent / f'spatial_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}.pkl'
        else:
            stat_fn = stat_fn.parent / f'spatial_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}.pkl'
    elif not spatial_masking and temporal_masking:
        if 'ratio' in temporal_mask_method:
            stat_fn = stat_fn.parent / f'temporal_{stat_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}.pkl'
        else:
            stat_fn = stat_fn.parent / f'temporal_{stat_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}.pkl'
    elif spatial_masking and temporal_masking:
        if 'ratio' in spatial_mask_method:
            if 'ratio' in temporal_mask_method:
                stat_fn = stat_fn.parent / (f'spatiotemporal_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.pkl')
            else:
                stat_fn = stat_fn.parent / (f'spatiotemporal_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.pkl')
        else:
            if 'ratio' in temporal_mask_method:
                stat_fn = stat_fn.parent / (f'spatiotemporal_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.pkl')
            else:
                stat_fn = stat_fn.parent / (f'spatiotemporal_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.pkl')
                
    return stat_fn


def compute_mahalanobis_ood_scores(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading train statistics...')
    stat_fn = get_statistics_filename('ucf101', ucf_fold, 'test', args)
    stat_fn = stat_fn.parent / f'{stat_fn.stem}_statistics_total.pkl'
    # stat_fn = '/home/simc/workspace/OOD-with-ViT/data/ucf101/statistics/masked/ucf101_1_train/spatiotemporal_max_0.5_lt_threshold_0.1_lt_threshold_0.05_statistics_total.pkl'
    # stat_fn = f'/home/simc/workspace/OOD-with-ViT/data/ucf101/statistics/ucf101_3_train_statistics_total.pkl'
    # stat_fn = 'data/ucf101/statistics/ucf101_1_train_statistics_total.pkl'
    
    
    print('train stat:', stat_fn)
    with open(stat_fn, 'rb') as f:
        train_stat_total = pickle.load(f)

    # prepare statistics
    means, precision = train_stat_total['mean'], train_stat_total['total_precision']
    for k400_class in means:
        means[k400_class] = torch.Tensor(means[k400_class]).to(device)
    precision = torch.Tensor(precision).to(device).float()

    # ucf101 vs. hmdb51
    # compute ucf101 mahalanobis distances
    print('loading ucf101 embeddings...')
    ucf101_embeddings_filename = get_embeddings_filename('ucf101', ucf_fold, 'test', args)
    # ucf101_embeddings_filename = '/home/simc/workspace/OOD-with-ViT/data/ucf101/embeddings/masked/ucf101_1_test/spatiotemporal_max_0.5_lt_threshold_0.01_lt_threshold_0.05.jsonl'
    print('ucf101:', ucf101_embeddings_filename)
    ucf101_embs = []
    with open(ucf101_embeddings_filename, 'r') as f:
        for line in tqdm(f):
            emb_js = json.loads(line)
            pre_logit = torch.Tensor(emb_js['penultimate'])
            ucf101_embs.append(pre_logit.view(1, -1))
    ucf101_embs = torch.cat(ucf101_embs, dim=0)

    print('computing k400 mahalanobis distances...')
    ucf101_gaussian_scores = []
    ucf101_features = ucf101_embs.to(device)
    for mean in tqdm(means.values()):
        zero_f = ucf101_features - mean
        gau_term = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
        ucf101_gaussian_scores.append(gau_term.cpu().view(-1, 1))
    ucf101_gaussian_scores = torch.cat(ucf101_gaussian_scores, dim=1)
    ucf101_mahalanobis_distances, _ = ucf101_gaussian_scores.min(dim=1)
    ucf101_mahalanobis_distances = ucf101_mahalanobis_distances.numpy().tolist()
        
    print('computing gaussian scores finished.')
    print('# of ucf101 embeddings:', len(ucf101_mahalanobis_distances))
        
    # compute k600 mahalanobis distances
    print('loading hmdb51 embeddings...')
    hmdb51_embeddings_filename = get_embeddings_filename('hmdb51', hmdb_fold, 'all', args)
    # hmdb51_embeddings_filename = '/home/simc/workspace/OOD-with-ViT/data/hmdb51/embeddings/masked/hmdb51_2_ucf101_1_train/spatiotemporal_max_0.5_lt_threshold_0.01_lt_threshold_0.05.jsonl'
    print('hmdb51:', hmdb51_embeddings_filename)
    hmdb51_embs = []
    with open(hmdb51_embeddings_filename, 'r') as f:
        for line in tqdm(f):
            emb_js = json.loads(line)
            pre_logit = torch.Tensor(emb_js['penultimate']).to(device)
            hmdb51_embs.append(pre_logit.view(1, -1))
    hmdb51_embs = torch.cat(hmdb51_embs, dim=0)

    print('computing hmdb51 mahalanobis distances...')
    hmdb51_gaussian_scores = []
    hmdb51_features = hmdb51_embs.to(device)
    for mean in tqdm(means.values()):
        zero_f = hmdb51_features - mean
        gau_term = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
        hmdb51_gaussian_scores.append(gau_term.cpu().view(-1, 1))
    print('computing gaussian scores finished.')
    hmdb51_gaussian_scores = torch.cat(hmdb51_gaussian_scores, dim=1)
    hmdb51_mahalanobis_distances, _ = hmdb51_gaussian_scores.min(dim=1)
    hmdb51_mahalanobis_distances = hmdb51_mahalanobis_distances.numpy().tolist()
    print('# of hmdb51 embeddings:', len(hmdb51_mahalanobis_distances))

    # min_len = min(len(ucf101_mahalanobis_distances), len(hmdb51_mahalanobis_distances))
    # ucf101_mahalanobis_distances = random.sample(ucf101_mahalanobis_distances, min_len)
    # hmdb51_mahalanobis_distances = random.sample(hmdb51_mahalanobis_distances, min_len)

    print('computing ucf101 vs. hmdb51 ood scores...')
    test_y = [0 for _ in range(len(ucf101_mahalanobis_distances))] + [1 for _ in range(len(hmdb51_mahalanobis_distances))]
    ood_scores = ucf101_mahalanobis_distances + hmdb51_mahalanobis_distances
    print('# of ucf101 and hmdb51 embeddings:', len(set(ood_scores)))

    measure_performances(ucf101_mahalanobis_distances, hmdb51_mahalanobis_distances)

    fpr, tpr, ucf_hmdb_auroc_score = auroc(test_y, ood_scores)
    pr, re, ucf_hmdb_aupr_score = aupr(test_y, ood_scores)
    ucf_hmdb_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
    result_filename = get_result_filename('ucf101', 'hmdb51', args)
    result_filename = result_filename.parent / (result_filename.stem + '_maha' + result_filename.suffix)
    with open(result_filename, 'w') as f:
        f.write(json.dumps({
            'auroc': ucf_hmdb_auroc_score,
            'aupr': ucf_hmdb_aupr_score,
            'fpr95': ucf_hmdb_fpr95,
        }))
    
    print('ucf101 vs. hmdb51:')
    print('auroc:', ucf_hmdb_auroc_score, 'aupr:', ucf_hmdb_aupr_score, 'fpr95:', ucf_hmdb_fpr95)
    image_filename = get_graph_filename('ucf101', 'hmdb51', args)
    auroc_filename = image_filename.parent / (image_filename.name + '_maha_auroc.png')
    save_roc_curve(fpr, tpr, auroc_filename)
    aupr_filename = image_filename.parent / (image_filename.name + '_maha_aupr.png')
    save_precision_recall_curve(pr, re, aupr_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--head_fusion', default='max')
    parser.add_argument('--discard_ratio', default=0.5)
    
    parser.add_argument('--spatial_masking', action='store_true')
    parser.add_argument('--spatial_mask_method', type=str)
    parser.add_argument('--spatial_mask_ratio', type=float)
    parser.add_argument('--spatial_mask_threshold', type=float)
    
    parser.add_argument('--temporal_masking', action='store_true')
    parser.add_argument('--temporal_mask_method', type=str)
    parser.add_argument('--temporal_mask_ratio', type=float)
    parser.add_argument('--temporal_mask_threshold', type=float)
    
    args = parser.parse_args()
    
    compute_mahalanobis_ood_scores(args)