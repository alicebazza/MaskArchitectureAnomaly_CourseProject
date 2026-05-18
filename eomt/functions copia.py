# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp

import importlib
import torch.nn.functional as F
from torch.amp import autocast
import matplotlib.pyplot as plt

from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize

from eval.erfnet import ERFNet

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3 # 3 canali RGB
NUM_CLASSES = 20
IGNORE_INDEX = 255


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
    
# crea modello ERFNet vuoto, carica pesi addestrati
def load_erfnet(args, device):
    erfnet_weightspath = osp.join(args.loadDir, args.erfnetWeights)
    # percorso del file dei pesi

    print("Loading ERFNet weights:", erfnet_weightspath)

    model = ERFNet(NUM_CLASSES).to(device)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(erfnet_weightspath, map_location=device)
    # carica il file dalla memoria
    checkpoint = extract_state_dict(checkpoint)
    # estrae solo i pesi del modello dal chechpoint

    model = load_my_state_dict(model, checkpoint) # copia i pesi dentro il modello
    model.eval()

    print("ERFNet loaded successfully")

    return model
    
# costruisce il modello a partire da una configurazione config, carica i pesi
# salvati da state_dict_path, sposta il modello su CPU/GPU
# e restituisce il modello pronto per inferenza
def load_eomt(device, config, state_dict_path):
    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=(1024, 1024), **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=19,
        encoder=encoder,
        **network_kwargs,
    )

    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = (
        lit_cls(
            img_size=(1024, 1024),
            num_classes=19,
            network=network,
            **model_kwargs,
        )
        .eval()
        .to(device)
    )

    if device == 'cpu':
        state_dict = torch.load(
                    state_dict_path, map_location="cpu", weights_only=True
                )
    else:
        state_dict = torch.load(
                    state_dict_path, map_location=f"cuda:{0}", weights_only=True
                )
    model.load_state_dict(state_dict, strict=False)
    print('Model\'s weights loaded succesfully')

    return model
    
# Combina le predizioni finali di maschere e classi per ottenere una mappa di logit per-pixel sulle classi
def eomt_to_pixel_logits(img, device, model):
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]
        # prende le ultime due dimensioni del tensore (H, W)
        
        crops, origins = model.window_imgs_semantic(imgs)
        # Divide l’immagine in finestre/crop più piccoli.
        # crops contiene i pezzi dell’immagine
        # origins contiene le posizioni originali dei crop nell’immagine completa.
    
        # forward del modello sui crop
        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], (1024, 1024), mode="bilinear"
        )
        
        # Combina: logits delle maschere e logits delle classi
        # per ottenere logits per ogni pixel di ciascun crop
        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        # Ricompone i logits dei vari crop nella forma dell’immagine originale
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

    return logits[0]


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
   

def load_ood_gt(path, size = None):
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
    
    
def plot_semantic_results(img, pred_array, target_array):
    mapping = create_mapping([pred_array, target_array], IGNORE_INDEX)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Image")
    axes[1].imshow(apply_colormap(pred_array, mapping))
    axes[1].set_title("Prediction")
    axes[2].imshow(apply_colormap(target_array, mapping))
    axes[2].set_title("Target")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
