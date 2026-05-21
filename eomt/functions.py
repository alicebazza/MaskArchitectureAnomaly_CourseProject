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
    
def eomt_to_pixel_logits_train(img, device, model):

    imgs = [img.to(device)]
    img_sizes = [img.shape[-2:] for img in imgs]

    crops, origins = model.window_imgs_semantic(imgs)

    # forward del modello sui crop
    mask_logits_per_layer, class_logits_per_layer = model(crops)

    mask_logits = F.interpolate(
        mask_logits_per_layer[-1],
        (1024, 1024),
        mode="bilinear",
        align_corners=False,
    )

    crop_logits = model.to_per_pixel_logits_semantic(
        mask_logits,
        class_logits_per_layer[-1],
    )

    logits = model.revert_window_logits_semantic(
        crop_logits,
        origins,
        img_sizes,
    )

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
   

def load_ood_gt(path, size=None):
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
    
    
def create_mapping(images, ignore_index):
    """
    Input:
        images: lista di immagini (matrici di pixel) da cui
          estrarre gli ID unici
        ignore_index: l'ID del pixel che rappresenta lo sfondo o una
          classe da ignorare.

    Output:
        mapping: un dizionario che mappa ogni ID univoco a un colore RGB
          espresso come array NumPy di 3 elementi. L'ID
          da ignorare viene mappato sul nero [0, 0, 0].
    """
    unique_ids = np.unique(np.concatenate([np.unique(img) for img in images]))
    valid_ids = unique_ids[unique_ids != ignore_index]

    colors = np.array(
        [plt.cm.hsv(i / len(valid_ids))[:3] for i in range(len(valid_ids))]
    )

    mapping = {cid: colors[i] for i, cid in enumerate(valid_ids)}
    mapping[ignore_index] = np.array([0, 0, 0])

    return mapping


def apply_colormap(image, mapping):
    """Prende un'immagine contenente ID di classe e la trasforma in un'immagine
    a colori RGB basandosi sul dizionario di mappatura fornito.
    """
    colored_image = np.zeros((*image.shape, 3))

    for cid in np.unique(image):
        colored_image[image == cid] = mapping.get(cid, [0, 0, 0])

    return colored_image
    

def plot_semantic_results_eomt(img, pred_array, target_array, save_path=None):
    """Visualizza e confronta l'immagine originale, la predizione di EoMTe la ground truth.

    Genera un grafico a tre pannelli (side-by-side) per valutare visivamente
    la qualità della segmentazione semantica. Permette sia di mostrare il
    grafico a schermo che di salvarlo su disco.

    Input:
        img: l'immagine originale in formato Tensor di PyTorch
        pred_array: la maschera di segmentazione predetta dal
          modello EomT
        target_array: la maschera di segmentazione reale (ground
          truth)
        save_path (str, optional): Il percorso in cui salvare l'immagine,
                                   se None, il grafico verrà mostrato a video.

    """
    mapping = create_mapping([pred_array, target_array], IGNORE_INDEX)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img.permute(1, 2, 0).detach().cpu().numpy())
    axes[0].set_title("Image")

    axes[1].imshow(apply_colormap(pred_array, mapping))
    axes[1].set_title("EoMT prediction")

    axes[2].imshow(apply_colormap(target_array, mapping))
    axes[2].set_title("OOD ground truth")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def freeze_model_except_final_parts(model):
    for param in model.parameters():
        param.requires_grad = False
        # scorre tutti i parametri del modello e li rende non trainabili

    for module in [model.q, model.class_head, model.mask_head], model.upscale:
        for param in module.parameters():
            param.requires_grad = True
            # riattivo questi parametri
            
    # q = query apprese che cercano oggetti/regione nell’immagine durante l’attention. Vanno allenate perché inizialmente non sanno cosa rappresentare nel tuo task.
    # class_head = trasforma ogni query nelle probabilità delle classi (num_classes + 1). Va allenata perché le classi del tuo dataset sono diverse da quelle del pretraining.
    # mask_head = converte le query in embedding usati per produrre le maschere di segmentazione. Va allenata perché deve imparare quali feature corrispondono alle regioni corrette.
    # upscale = aumenta la risoluzione delle feature del ViT per ottenere maschere più dettagliate. Va allenato perché è un modulo nuovo e deve imparare a ricostruire dettagli spaziali.

    print_trainable_parameters(model)


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
    # scorre tutti i parametri del modello
        total += param.numel()
        if param.requires_grad:
        # controlla se il parametro è trainabile
            trainable += param.numel()
            print(name)

    perc = 100.0 * trainable / total
    print(f"\nTrainable params: {trainable}/{total} = {perc:.4f}%\n")


def ood_hinge_loss(logits, ood_mask, margin=0.1):
    """
    logits:   [19, H, W] oppure [1, 19, H, W]
    ood_mask: [H, W] oppure [1, H, W]
    """

    if logits.dim() == 3:
        logits = logits.unsqueeze(0)

    if ood_mask.dim() == 2:
        ood_mask = ood_mask.unsqueeze(0)

    ood_mask = ood_mask.to(logits.device).bool()

    probs = torch.sigmoid(logits)
    # trasformo in probabilita non esclusive (sigmoid)
    confidence = probs.sum(dim=1)
    # somma le probbailità sulle 19 classi note
    # [B, 19, H, W] -> [B, H, W]

    loss_map = F.relu(confidence - margin) ** 2

    if not ood_mask.any():
        return logits.new_tensor(0.0)
    # se non ci sono pixel OoD nella maschera restituisce 0

    return loss_map[ood_mask].mean()
