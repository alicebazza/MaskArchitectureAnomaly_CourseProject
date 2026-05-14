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


def load_my_state_dict(model, state_dict):
    """
    Carica manualmente i pesi (state_dict) in un modello PyTorch esistente.

    Input:
        model (torch.nn.Module): modello già istanziato (architettura definita)
        state_dict (dict): dizionario dei pesi da caricare (nome -> tensore)

    Output:
        model (torch.nn.Module): modello con i pesi aggiornati
    """
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

def extract_state_dict(checkpoint):
    """
    Estrae lo state_dict da un checkpoint salvato in formati diversi.
    Supporta checkpoint con diverse chiavi

    Input:
        checkpoint (dict): oggetto caricato da torch.load()

    Output:
        state_dict (dict): dizionario dei pesi (nome -> tensore)
    """
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


# più il modello è incerto ---> più probabile che ci sia un'anomalia
def anomaly_scores(logits, use_rba=False):
    """
    Calcola diverse mappe di anomaly score a partire dai logits per pixel.

    Score calcolati:
        - MSP (Maximum Softmax Probability)
        - MaxLogit
        - Entropy (normalizzata)
        - RBA (opzionale)

    Input:
        logits (torch.Tensor): tensore di dimensione (C, H, W) contenente i logits
            (output grezzo della rete, prima della softmax)
        use_rba (bool): se True, calcola anche lo score RBA

    Output:
        scores (list of torch.Tensor): lista di mappe (H, W), una per ogni anomaly score
    """

    # probabilità tramite softmax sui logits
    probs = torch.softmax(logits, dim=0)

    scores = []

    # MSP: 1 - max probabilità
    msp = 1.0 - torch.max(probs, dim=0)[0]
    scores.append(msp)

    # MaxLogit: negativo del logit massimo
    maxlogit = -torch.max(logits, dim=0)[0]
    scores.append(maxlogit)

    # Entropy normalizzata
    K = probs.shape[0]
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=0)
    entropy = entropy / torch.log(torch.tensor(float(K), device=probs.device))
    scores.append(entropy)

    # RBA opzionale
    if use_rba:
        rba = -torch.tanh(logits).sum(dim=0)
        scores.append(rba)

    return scores
   

def load_ood_gt(path):
    """
    Carica la maschera ground truth (OOD) a partire dal percorso dell'immagine.
    Costruisce automaticamente il path della maschera e applica trasformazioni
    specifiche a seconda del dataset

    Input:
        path (str): percorso dell'immagine di input

    Output:
        ood_gts (np.ndarray): maschera OOD come array numpy
    """
    # parte dal path dell'immagine e trova automatica la maschera corrispondente
    pathGT = path.replace("images", "labels_masks")

    if "RoadObstacle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")

    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    mask = Image.open(pathGT)

    target_transform = Compose([
        Resize(size, Image.NEAREST),
    ])

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


def eval_score(ood_gts_list, anomaly_score_list):
    """
    Valuta le mappe di anomaly score confrontandole con le maschere ground truth OOD.
    Estrae separatamente gli score sui pixel normali e anomali, costruisce le etichette
    binarie corrispondenti e calcola le metriche AP/AUPRC e FPR@95TPR.

    Input:
        ood_gts_list: lista di maschere ground truth OOD,
            con valori 0 = in-distribution e 1 = OOD
        anomaly_score_list: lista di mappe di anomaly score,
            una per immagine, con dimensioni compatibili con le maschere

    Output:
        prc_auc (float): Average Precision / area sotto la Precision-Recall curve
        fpr (float): false positive rate quando il true positive rate è al 95%
    """
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
