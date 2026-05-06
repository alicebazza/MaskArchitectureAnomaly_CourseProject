# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # per guardare sia eval che eomt
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3 # 3 canali RGB
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# pre-processing per le immagini di input
input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

# pre-processing maschere groud-truth
target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

# prende il modello già istanziato (architettura definita)
# e lo state_dict già addestrato
# carica manualmente i pesi dentro un modello esistente
def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model

# serve a estrarre lo state_dict da un checkpoint
def extract_state_dict(checkpoint):
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


# calcola diversi punteggi anomalia
# più il modello è incerto ---> più probabile che ci sia un'anomalia
def anomaly_scores(tensor, use_rba = False, is_probs = False):
    scores = []

    if is_probs:
        # Se in input abbiamo già probabilità (EoMT), le usiamo direttamente
        probs = tensor
        logits_for_max = torch.log(probs + 1e-8)
    else:
        # Se in input abbiamo logit (ERFNet), calcoliamo la softmax
        probs = torch.softmax(tensor, dim=0)
        logits_for_max = tensor

    msp = 1.0 - torch.max(probs, dim=0)[0] # modello incerto ---> valore alto
    scores.append(msp)

    maxlogit = -torch.max(logits_for_max, dim=0)[0]
    scores.append(maxlogit)

    K = probs.shape[0]
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=0)
    entropy = entropy / torch.log(torch.tensor(float(K), device=probs.device))
    scores.append(entropy)

    if use_rba:
        rba = -torch.tanh(logits_for_max).sum(dim=0)
        scores.append(rba)

    return scores
   
# prende in input il percorso di un'immagine e restituisce la maschera
def load_ood_gt(path):
    pathGT = path.replace("images", "labels_masks")
    # costruisce il percorso della maschera ground truth
    # assume che maschera e immagine abbiano lo stesso nome ma in cartelle diverse

    if "RoadObstacle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")

    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    mask = Image.open(pathGT)
    mask = target_transform(mask)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in pathGT:
        ood_gts = np.where(ood_gts == 2, 1, ood_gts)

    if "LostAndFound" in pathGT:
        ood_gts = np.where(ood_gts == 0, 255, ood_gts)
        ood_gts = np.where(ood_gts == 1, 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

    if "Streethazard" in pathGT:
        ood_gts = np.where(ood_gts == 14, 255, ood_gts)
        ood_gts = np.where(ood_gts < 20, 0, ood_gts)
        ood_gts = np.where(ood_gts == 255, 1, ood_gts)

    return ood_gts

# prende in input due liste: una di maschere e una di punteggi anomalia (una per immagine)
# e restituisce AURPC e FPR95TPR
def eval_score(ood_gts_list, anomaly_score_list):
    ood_gts = np.array(ood_gts_list) # dim (N,H,W) con N = numero di immagini
    anomaly_scores = np.array(anomaly_score_list)
        
    ood_mask = (ood_gts == 1) # true sui pixel OoD
    ind_mask = (ood_gts == 0) # true sui pixel in-distribution

    ood_out = anomaly_scores[ood_mask] # score su pixel OoD
    ind_out = anomaly_scores[ind_mask] # score su pixel normali

    ood_label = np.ones(len(ood_out)) # etichette vere OoD = 1
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    return prc_auc, fpr
