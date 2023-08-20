import matplotlib.pyplot as plt
import numpy as np



def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
    
def save_roc_curve(fpr, tpr, path):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig(path)
    plt.clf()
    
def plot_precision_recall_curve(precision, recall):
    plt.plot(recall, precision, color='red', label='PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
def save_precision_recall_curve(precision, recall, path):
    plt.plot(recall, precision, color='red', label='PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(path)
    plt.clf()
    
def save_histogram(id, ood, id_label, ood_label, detector, path, auc=0.0):
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.xmargin'] = 0
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # plt.rcParams['axes.ymargin'] = 0
    
    def kl_divergence(p, q):
        eps = 1e-5
        p, q = p / p.sum(), q / q.sum()
        p, q = p + eps, q + eps
        return np.sum(p * np.log(p / q))
    
    id, ood = np.array(id), np.array(ood)
    if 'Energy' in detector or 'MSP' in detector:
        id, ood = -id, -ood
    max_ = max(id.max(), ood.max())
    min_ = min(id.min(), ood.min())
    if 'Energy' in detector or 'MSP' in detector:
        id, ood = id / max_, ood / max_
        id, ood = 1 - id, 1 - ood
    else:
        id, ood = (id - min_) / max_, (ood - min_) / max_
    
    id_hist, _ = np.histogram(id, bins=100)
    ood_hist, _ = np.histogram(ood, bins=100)
    kld = kl_divergence(id_hist, ood_hist)
    print('kl divergence:', kld)
    
    id_label, ood_label = f'in-distribution ({id_label})', f'out-of-distribution ({ood_label})'
    path = path.replace('png', 'pdf')
    fig, ax = plt.subplots()
    plt.hist(id, histtype='bar', range=[-0., 1.01,], bins=60, alpha=0.7, label=id_label, rwidth=0.4, density=True, color='blue')
    plt.hist(ood, histtype='bar', range=[-0., 1.01,], bins=50, alpha=0.7, label=ood_label, rwidth=0.4, density=True, color='red')
    # plt.hist(id, histtype='bar', range=[0.1, 0.4,], bins=110, alpha=0.7, label=id_label, rwidth=0.3, density=True, color='blue')
    # plt.hist(ood, histtype='bar', range=[0.1, 0.4,],  bins=100, alpha=0.7, label=ood_label, rwidth=0.3, density=True, color='red')
    plt.xlabel(f'{detector} uncertainty')
    plt.ylabel('Density')
    # plt.legend()
    plt.rcParams['font.size'] = 25
    plt.text(0.47, 0.90, f'AUROC = {auc:.2f}', transform=ax.transAxes)
    plt.text(0.47, 0.78, f'KLD = {kld:.3f}', transform=ax.transAxes)
    plt.xticks([0., 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.ylim(0., 4.9)
    # plt.ylim(0., 16.0)
    plt.ylim(0., 9.)
    plt.savefig(path, bbox_inches='tight')
    plt.clf()
    