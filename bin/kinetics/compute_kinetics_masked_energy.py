import json
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import random

from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.utils.visualization import save_roc_curve, save_precision_recall_curve, save_histogram

random.seed(1234)

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
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
        else:
            if 'ratio' in temporal_mask_method:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
                
    return result_fn

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
   
def get_embeddings_filename(
    dataset: str,
    split: str,
    args,
):
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    
    embeddings_dir = Path('./data') / 'kinetics' / 'embeddings' / 'masked' / f'{dataset}_{split}'
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
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
                
    return emb_fn

def compute_energy_ood_scrores(args):
    # kinetics400 vs. kinetics600
    # compute k400 mahalanobis distances
    T = args.T
    
    print('loading k400 embeddings...')
    k400_embeddings_filename = get_embeddings_filename('k400', 'val', args)
    print('k400:', k400_embeddings_filename)
    k400_val = []
    with open(k400_embeddings_filename, 'r') as f:
        for line in tqdm(f):
            emb_js = json.loads(line)
            logit = torch.Tensor(emb_js['logit'])
            k400_val.append(logit.view(1, -1))
    k400_logits = torch.cat(k400_val, dim=0)
    k400_energy = T * torch.logsumexp(k400_logits / T, dim=1)
    k400_energy = (-k400_energy.numpy()).tolist()
        
    # compute k600 mahalanobis distances
    print('loading k600 embeddings...')
    k600_emb_fn = get_embeddings_filename('k600', 'val', args)
    k600_embeddings_filename = k600_emb_fn.parent / (k600_emb_fn.stem + '_deduplicated' + k600_emb_fn.suffix)
    k600_original = []
    with open(k600_embeddings_filename, 'r') as f:
        for line in tqdm(f):
            emb_js = json.loads(line)
            logit = torch.Tensor(emb_js['logit'])
            k600_original.append(logit.view(1, -1))
    k600_logits = torch.cat(k600_original, dim=0)
    k600_energy = T * torch.logsumexp(k600_logits / T, dim=1)
    k600_energy = (-k600_energy.numpy()).tolist()

    # min_len = min(len(k400_msp), len(k600_msp))
    # k400_msp = random.sample(k400_msp, min_len)
    # k600_msp = random.sample(k600_msp, min_len)
    
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
        path=f'histograms/{id_label}+{ood_label}_{detector}_masked.png',
        auc=75.95
    )

    fpr, tpr, k600_auroc_score = auroc(test_y, ood_scores)
    pr, re, k600_aupr_score = aupr(test_y, ood_scores)
    k600_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
    result_filename = get_result_filename('k400', 'k600', args)
    result_filename = result_filename.parent / (result_filename.stem + '_msp' + result_filename.suffix)
    with open(result_filename, 'w') as f:
        f.write(json.dumps({
            'auroc': k600_auroc_score,
            'aupr': k600_aupr_score,
            'fpr95': k600_fpr95,
        }))
    
    print('auroc:', k600_auroc_score, 'aupr:', k600_aupr_score, 'fpr95:', k600_fpr95)

    image_filename = get_graph_filename('k400', 'k600', args)
    auroc_filename = image_filename.parent / (image_filename.name + '_msp_auroc.png')
    save_roc_curve(fpr, tpr, auroc_filename)
    aupr_filename = image_filename.parent / (image_filename.name + '_msp_aupr.png')
    save_precision_recall_curve(pr, re, aupr_filename)

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

    # min_len = min(len(k400_msp), len(k700_msp))
    # k400_msp = random.sample(k400_msp, min_len)
    # k700_msp = random.sample(k700_msp, min_len)

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
        path=f'histograms/{id_label}+{ood_label}_{detector}_masked.png',
        auc=78.02
    )

    fpr, tpr, k700_auroc_score = auroc(test_y, ood_scores)
    pr, re, k700_aupr_score = aupr(test_y, ood_scores)
    k700_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
    result_filename = get_result_filename('k400', 'k700', args)
    result_filename = result_filename.parent / (result_filename.stem + '_msp' + result_filename.suffix)
    
    print('k400 vs. k700:')
    print('auroc:', k700_auroc_score, 'aupr:', k700_aupr_score, 'fpr95:', k700_fpr95)
    with open(result_filename, 'w') as f:
        f.write(json.dumps({
            'auroc': k700_auroc_score,
            'aupr': k700_aupr_score,
            'fpr95': k700_fpr95,
        }))
    image_filename = get_graph_filename('k400', 'k700', args)
    auroc_filename = image_filename.parent / (image_filename.name + '_msp_auroc.png')
    save_roc_curve(fpr, tpr, auroc_filename)
    aupr_filename = image_filename.parent / (image_filename.name + '_msp_aupr.png')
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
    
    parser.add_argument('--T', default=1.0, type=float)
    
    args = parser.parse_args()
    
    compute_energy_ood_scrores(args)