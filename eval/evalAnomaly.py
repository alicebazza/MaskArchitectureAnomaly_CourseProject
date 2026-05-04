# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from eomt.models.eomt import EoMT
from eomt.models.vit import ViT
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

# crea modello ERFNet vuoto, carica pesi addestrati
def load_erfnet(args, device):
    erfnet_weightspath = osp.join(args.loadDir, args.erfnetWeights)
    # percorso del file dei pesi

    print("Loading ERFNet weights:", erfnet_weightspath)

    model = ERFNet(NUM_CLASSES).to(device)

    if not args.cpu:
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(erfnet_weightspath, map_location=device)
    # carica il file dalla memoria
    checkpoint = extract_state_dict(checkpoint)
    # estrae solo i pesi del modello dal chechpoint

    model = load_my_state_dict(model, checkpoint) # copia i pesi dentro il modello
    model.eval()

    print("ERFNet loaded successfully")

    return model

def load_eomt(args, device):
    eomt_weightspath = osp.join(args.loadDir, args.eomtWeights)

    print("Loading EoMT weights:", eomt_weightspath)

    encoder = ViT(
        img_size=(512, 1024),
        patch_size=14,
        backbone_name="vit_large_patch14_reg4_dinov2",
    )

    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100,
        num_blocks=4,
        masked_attn_enabled=True,
    )
    
    model = model.to(device)
    
    if not args.cpu:
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(eomt_weightspath, map_location=device)
    checkpoint = extract_state_dict(checkpoint)

    model = load_my_state_dict(model, checkpoint)
    model.eval()

    print("EoMT loaded successfully")

    return model

# calcola diversi punteggi anomalia
def anomaly_scores(logits, use_rba=False):
    scores = []

    probs = torch.softmax(logits, dim=0)

    msp = 1.0 - torch.max(probs, dim=0)[0]
    scores.append(msp)

    maxlogit = -torch.max(logits, dim=0)[0]
    scores.append(maxlogit)

    K = probs.shape[0]
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=0)
    entropy = entropy / torch.log(torch.tensor(float(K), device=probs.device))
    scores.append(entropy)

    if use_rba:
        rba = -torch.tanh(logits).sum(dim=0)
        scores.append(rba)

    return scores

# Combina le predizioni finali di maschere e classi (per query) per ottenere una mappa di probabilità per-pixel sulle classi
# Restituisce le log-probabilità per pixel (C × H × W), normalizzate sulle classi.
def eomt_to_pixel_logits(mask_logits_per_layer, class_logits_per_layer):
    mask_logits = mask_logits_per_layer[-1]
    class_logits = class_logits_per_layer[-1]

    # porta le maschere alla risoluzione finale
    mask_logits = torch.nn.functional.interpolate(
        mask_logits,
        size=(512, 1024),
        mode="bilinear",
        align_corners=False,
    )

    mask_prob = torch.sigmoid(mask_logits)
    class_prob = torch.softmax(class_logits, dim=-1)

    class_prob = class_prob[:, :, :-1]

    pixel_probs = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
    pixel_probs = pixel_probs / (pixel_probs.sum(dim=1, keepdim=True) + 1e-8)

    probs = pixel_probs.squeeze(0)
    logits = torch.log(probs + 1e-8) # DA CAMBIARE !!

    return logits
    
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
    ood_gts = np.array(ood_gts_list) # dim (N,H,W) con N=numero di immagini
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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObstacle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--erfnetWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--eomtWeights', default="eomt_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    # liste vuote dove verranno salvati i punteggi anomalia
    anomaly_score_msp_list_ERFNet = []
    anomaly_score_maxlogit_list_ERFNet = []
    anomaly_score_maxentropy_list_ERFNet = []
    anomaly_score_msp_list_EoMT = []
    anomaly_score_maxlogit_list_EoMT = []
    anomaly_score_maxentropy_list_EoMT = []
    anomaly_score_rba_list_EoMT = []
    
    ood_gts_list = [] # maschere ground truth OoD

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')
    
    device = torch.device("cpu" if args.cpu else "cuda")
    
    # carica i due modelli
    model_ERFNet = load_erfnet(args, device)
    model_EoMT = load_eomt(args, device)
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
    # ciclo su tutte le immagini
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        # images = images.permute(0,3,1,2)
        with torch.no_grad():
            # ERFNet inference
            result_ERFNet = model_ERFNet(images)
            logits_ERFNet = result_ERFNet.squeeze(0)

            # EoMT inference
            mask_logits_per_layer, class_logits_per_layer = model_EoMT(images)

            logits_EoMT = eomt_to_pixel_logits(
                mask_logits_per_layer,
                class_logits_per_layer
            )
            
        # anomaly scores
        scores_ERFNet = anomaly_scores(logits_ERFNet, use_rba=False)
        scores_EoMT = anomaly_scores(logits_EoMT, use_rba=True)

        # ground truth OOD
        ood_gts = load_ood_gt(path)

        # salta immagini senza pixel OOD
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        
        # ERFNet
        anomaly_score_msp_list_ERFNet.append(
            scores_ERFNet[0].cpu().numpy()
        )
        anomaly_score_maxlogit_list_ERFNet.append(
            scores_ERFNet[1].cpu().numpy()
        )
        anomaly_score_maxentropy_list_ERFNet.append(
            scores_ERFNet[2].cpu().numpy()
        )

        # EoMT
        anomaly_score_msp_list_EoMT.append(
            scores_EoMT[0].cpu().numpy()
        )
        anomaly_score_maxlogit_list_EoMT.append(
            scores_EoMT[1].cpu().numpy()
        )
        anomaly_score_maxentropy_list_EoMT.append(
            scores_EoMT[2].cpu().numpy()
        )
        anomaly_score_rba_list_EoMT.append(
            scores_EoMT[3].cpu().numpy()
        )
        
        del images
        del result_ERFNet
        del mask_logits_per_layer
        del class_logits_per_layer
        del logits_ERFNet
        del logits_EoMT
        del scores_ERFNet
        del scores_EoMT
        del ood_gts

        if device.type == "cuda":
            torch.cuda.empty_cache()


    file.write( "\n")

    # evaluation ERFNet
    prc_auc_msp_ERFNet, fpr_msp_ERFNet = eval_score(
        ood_gts_list,
        anomaly_score_msp_list_ERFNet
    )

    prc_auc_maxlogit_ERFNet, fpr_maxlogit_ERFNet = eval_score(
        ood_gts_list,
        anomaly_score_maxlogit_list_ERFNet
    )

    prc_auc_maxentropy_ERFNet, fpr_maxentropy_ERFNet = eval_score(
        ood_gts_list,
        anomaly_score_maxentropy_list_ERFNet
    )
    
    
    # evaluation EoMT
    prc_auc_msp_EoMT, fpr_msp_EoMT = eval_score(
        ood_gts_list,
        anomaly_score_msp_list_EoMT
    )

    prc_auc_maxlogit_EoMT, fpr_maxlogit_EoMT = eval_score(
        ood_gts_list,
        anomaly_score_maxlogit_list_EoMT
    )

    prc_auc_maxentropy_EoMT, fpr_maxentropy_EoMT = eval_score(
        ood_gts_list,
        anomaly_score_maxentropy_list_EoMT
    )

    prc_auc_rba_EoMT, fpr_rba_EoMT = eval_score(
        ood_gts_list,
        anomaly_score_rba_list_EoMT
    )
    
    # stampa ERFNet
    print(f"AUPRC msp score ERFNet: {prc_auc_msp_ERFNet * 100.0}")
    print(f"FPR@TPR95 msp ERFNet: {fpr_msp_ERFNet * 100.0}")

    print(f"AUPRC maxlogit score ERFNet: {prc_auc_maxlogit_ERFNet * 100.0}")
    print(f"FPR@TPR95 maxlogit ERFNet: {fpr_maxlogit_ERFNet * 100.0}")

    print(f"AUPRC maxentropy score ERFNet: {prc_auc_maxentropy_ERFNet * 100.0}")
    print(f"FPR@TPR95 maxentropy ERFNet: {fpr_maxentropy_ERFNet * 100.0}")

    # stampa EoMT
    print(f"AUPRC msp score EoMT: {prc_auc_msp_EoMT * 100.0}")
    print(f"FPR@TPR95 msp EoMT: {fpr_msp_EoMT * 100.0}")

    print(f"AUPRC maxlogit score EoMT: {prc_auc_maxlogit_EoMT * 100.0}")
    print(f"FPR@TPR95 maxlogit EoMT: {fpr_maxlogit_EoMT * 100.0}")

    print(f"AUPRC maxentropy score EoMT: {prc_auc_maxentropy_EoMT * 100.0}")
    print(f"FPR@TPR95 maxentropy EoMT: {fpr_maxentropy_EoMT * 100.0}")

    print(f"AUPRC rba score EoMT: {prc_auc_rba_EoMT * 100.0}")
    print(f"FPR@TPR95 rba EoMT: {fpr_rba_EoMT * 100.0}")
    
    # scrittura su file
    file.write(
        "ERFNet\n"
        f"AUPRC msp score ERFNet: {prc_auc_msp_ERFNet * 100.0} "
        f"FPR@TPR95 msp ERFNet: {fpr_msp_ERFNet * 100.0}\n"
        f"AUPRC maxlogit score ERFNet: {prc_auc_maxlogit_ERFNet * 100.0} "
        f"FPR@TPR95 maxlogit ERFNet: {fpr_maxlogit_ERFNet * 100.0}\n"
        f"AUPRC maxentropy score ERFNet: {prc_auc_maxentropy_ERFNet * 100.0} "
        f"FPR@TPR95 maxentropy ERFNet: {fpr_maxentropy_ERFNet * 100.0}\n\n"
    )

    file.write(
        "EoMT\n"
        f"AUPRC msp score EoMT: {prc_auc_msp_EoMT * 100.0} "
        f"FPR@TPR95 msp EoMT: {fpr_msp_EoMT * 100.0}\n"
        f"AUPRC maxlogit score EoMT: {prc_auc_maxlogit_EoMT * 100.0} "
        f"FPR@TPR95 maxlogit EoMT: {fpr_maxlogit_EoMT * 100.0}\n"
        f"AUPRC maxentropy score EoMT: {prc_auc_maxentropy_EoMT * 100.0} "
        f"FPR@TPR95 maxentropy EoMT: {fpr_maxentropy_EoMT * 100.0}\n"
        f"AUPRC rba score EoMT: {prc_auc_rba_EoMT * 100.0} "
        f"FPR@TPR95 rba EoMT: {fpr_rba_EoMT * 100.0}\n"
    )
    
    file.close() # scriviamo su result.txt

if __name__ == '__main__':
    main()
