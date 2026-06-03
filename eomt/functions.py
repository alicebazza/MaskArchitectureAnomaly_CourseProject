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
from typing import Any

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


def load_eomt(device: str | torch.device, config: dict[str, Any], state_dict_path: str) -> torch.nn.Module:
    """
    Input:
        device: dispositivo su cui caricare il modello.
        config: dizionario di configurazione del modello.
        state_dict_path: percorso del file contenente i pesi salvati.

    Output:
        model: modello caricato, in modalità eval e spostato sul device scelto.

    Cosa fa:
        Costruisce encoder, network e modulo Lightning a partire dalla configurazione,
        carica i pesi dal file indicato e restituisce il modello pronto per inferenza.
    """
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
    
def eomt_to_pixel_logits(img: torch.Tensor, device: str | torch.device, model: torch.nn.Module) -> torch.Tensor:
    """
    Input:
        img: immagine di input come tensore PyTorch.
        device: dispositivo su cui eseguire l'inferenza.
        model: modello EoMT già caricato.

    Output:
        logits: tensore dei logits per-pixel con dimensione (C, H, W).

    Cosa fa:
        Divide l'immagine in crop, applica il modello, combina logits di maschere
        e classi, poi ricompone i logits nella dimensione originale dell'immagine.
    """
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
def anomaly_scores(logits: torch.Tensor, use_rba: bool = False) -> list[torch.Tensor]:
    """
    Input:
        logits: tensore di dimensione (C, H, W) contenente i logits per-pixel.
        use_rba: se True, aggiunge anche lo score RBA alla lista degli score.

    Output:
        scores: lista di tensori di dimensione (H, W), uno per ogni anomaly score.

    Cosa fa:
        Calcola diverse mappe di anomaly score a partire dai logits: MSP,
        MaxLogit, entropia normalizzata e, opzionalmente, RBA.
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
   

def load_ood_gt(path: str, size: tuple[int, int] | int | None = None) -> np.ndarray:
    """
    Input:
        path: percorso dell'immagine di input.
        size: dimensione finale della maschera dopo il resize.

    Output:
        ood_gts: maschera OOD come array NumPy.

    Cosa fa:
        Costruisce automaticamente il percorso della maschera ground truth a partire
        dal percorso dell'immagine, la carica, la ridimensiona e normalizza le label
        in base al dataset.
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


def eval_score(ood_gts_list: list[np.ndarray], anomaly_score_list: list[np.ndarray]) -> tuple[float, float]:
    """
    Input:
        ood_gts_list: lista di maschere ground truth OOD, con valori
            0 = in-distribution e 1 = OOD.
        anomaly_score_list: lista di mappe di anomaly score, una per immagine,
            con dimensioni compatibili con le maschere.

    Output:
        prc_auc: Average Precision, cioè area sotto la curva Precision-Recall.
        fpr: false positive rate quando il true positive rate è al 95%.

    Cosa fa:
        Estrae gli score sui pixel normali e anomali, costruisce le etichette
        binarie corrispondenti e calcola AP/AUPRC e FPR@95TPR.
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
    
  
