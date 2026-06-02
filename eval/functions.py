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
from IPython.display import display

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


def load_my_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    """
    Input:
        model: modello PyTorch gia istanziato.
        state_dict: dizionario dei pesi da caricare, con nome del parametro
            come chiave e tensore come valore.

    Output:
        model: modello PyTorch con i pesi aggiornati.

    Cosa fa:
        Carica manualmente i pesi in un modello PyTorch, gestendo anche il
        caso in cui i nomi dei parametri siano preceduti da "module.".
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


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    Input:
        checkpoint: dizionario caricato da torch.load(), eventualmente
            contenente uno state_dict sotto una chiave specifica.

    Output:
        state_dict: dizionario dei pesi del modello, con nome del parametro
            come chiave e tensore come valore.

    Cosa fa:
        Estrae lo state_dict da checkpoint salvati in formati diversi,
        controllando prima le chiavi "state_dict" e "model".
    """
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint
    

def load_erfnet(args: Any, device: torch.device) -> torch.nn.Module:
    """
    Input:
        args: oggetto contenente i parametri del programma, in particolare
            loadDir ed erfnetWeights.
        device: dispositivo PyTorch su cui caricare il modello, CPU o CUDA.

    Output:
        model: modello ERFNet pronto per l'inferenza con i pesi caricati.

    Cosa fa:
        Istanzia ERFNet, carica i pesi da checkpoint, sposta il modello sul
        dispositivo richiesto e lo imposta in modalita di valutazione.
    """
    erfnet_weightspath = osp.join(args.loadDir, args.erfnetWeights)
    # percorso del file dei pesi

    print("Loading ERFNet weights:", erfnet_weightspath)

    model = ERFNet(NUM_CLASSES).to(device)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(erfnet_weightspath, map_location=device)
    # carica il file dalla memoria
    checkpoint = extract_state_dict(checkpoint)
    # estrae solo i pesi del modello dal checkpoint

    model = load_my_state_dict(model, checkpoint) # copia i pesi dentro il modello
    model.eval()

    print("ERFNet loaded successfully")

    return model
    

# più il modello è incerto ---> più probabile che ci sia un'anomalia
def anomaly_scores(
    logits: torch.Tensor,
    use_rba: bool = False,
) -> list[torch.Tensor]:
    """
    Input:
        logits: tensore di dimensione (C, H, W) contenente i logits per pixel.
        use_rba: booleano che indica se calcolare anche lo score RBA.

    Output:
        scores: lista di tensori di dimensione (H, W), uno per ogni anomaly
            score calcolato.

    Cosa fa:
        Calcola mappe di anomaly score a partire dai logits: MSP, MaxLogit,
        Entropy normalizzata e, opzionalmente, RBA.
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
   

def load_ood_gt(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """
    Input:
        path: percorso dell'immagine di input.
        size: dimensione a cui ridimensionare la maschera, oppure None.

    Output:
        ood_gts: maschera OOD come array NumPy.

    Cosa fa:
        Ricava il percorso della maschera ground truth a partire dal percorso
        dell'immagine, la carica, la ridimensiona e normalizza le etichette in
        base al dataset.
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


def eval_score(
    ood_gts_list: list[np.ndarray],
    anomaly_score_list: list[np.ndarray],
) -> tuple[float, float]:
    """
    Input:
        ood_gts_list: lista di maschere ground truth OOD, con valori
            0 = in-distribution e 1 = OOD.
        anomaly_score_list: lista di mappe di anomaly score, una per immagine,
            con dimensioni compatibili con le maschere.

    Output:
        prc_auc: Average Precision, cioe area sotto la curva Precision-Recall.
        fpr: false positive rate quando il true positive rate e al 95%.

    Cosa fa:
        Confronta gli anomaly score con le maschere OOD, costruisce le etichette
        binarie pixel-wise e calcola AP/AUPRC e FPR@95TPR.
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
    """Crea una mappatura univoca di colori RGB per ogni ID di classe presente
    nelle immagini.

    Input:
        images: una lista di array NumPy rappresentanti le
          immagini di segmentazione.
        ignore_index: l'ID della classe da ignorare, che
          verrà mappato sul colore nero.

    Output:
        dict: un dizionario dove le chiavi sono gli ID delle classi
              e i valori sono i rispettivi colori RGB.
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
    

def plot_semantic_results_erfnet(img, pred_array, target_array, save_path=None):
    """Visualizza e confronta l'immagine originale, la predizione di ERFNet e la ground truth.

    Genera un grafico a tre pannelli (side-by-side) per valutare visivamente
    la qualità della segmentazione semantica. Permette sia di mostrare il
    grafico a schermo che di salvarlo su disco.

    Input:
        img: l'immagine originale in formato Tensor di PyTorch
        pred_array: la maschera di segmentazione predetta dal
          modello ERFNet
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
    axes[1].set_title("ERFNet prediction")

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
