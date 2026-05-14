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
from eomt.models.eomt import EoMT
from eomt.models.vit import ViT
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from eval.evalAnomaly import *

# pre-processing per le immagini di input
input_transform = Compose(
    [
        Resize((1024, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

# pre-processing maschere groud-truth
target_transform = Compose(
    [
        Resize((1024, 1024), Image.NEAREST),
    ]
)

def load_eomt(args, device, config=None):
    # 1. Prendi il nome del modello
    name = getattr(args, "eomtName", None)

    if name is None and config is not None:
        name = (
            config.get("trainer", {})
            .get("logger", {})
            .get("init_args", {})
            .get("name")
        )

    if name is None:
        raise ValueError(
            "Nome modello EoMT mancante. Passa --eomtName oppure mettilo nel config."
        )

    # encoder ViT ---> estrae feature
    encoder = ViT(
        img_size=(1024, 1024),
        patch_size=14,
        backbone_name="vit_base_patch14_reg4_dinov2",
    )

    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100, # cerca fino a 100 oggetti diversi per ogni immagine
        num_blocks=3, # usiamo gli ultimi 3 blocchi del Transformer
        masked_attn_enabled=True, # limita l'attenzione delle query solo alle regioni dove è stata inizialmente trovata una maschera
    ).to(device)
    
    # 4. Scarica pesi
    state_dict_path = "/content/drive/MyDrive/eomt_cityscapes.bin"

    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Non trovo il file su Drive Percorso cercato: {state_dict_path}")

    # 5. Carica pesi
    checkpoint = torch.load(
        state_dict_path,
        map_location=device,
    )
    checkpoint = extract_state_dict(checkpoint)
    model = load_my_state_dict(model, checkpoint)

    model.eval()

    print("EoMT loaded successfully")

    return model

# Combina le predizioni finali di maschere e classi (per query) per ottenere una mappa di logit per-pixel sulle classi
def eomt_to_pixel_logits(mask_logits_per_layer, class_logits_per_layer):
    mask_logits = mask_logits_per_layer[-1] # prendiamo solo l'output finale
    class_logits = class_logits_per_layer[-1]

    # porta le maschere alla risoluzione finale per farle combaciare con l'immagine di input
    mask_logits = torch.nn.functional.interpolate(
        mask_logits,
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
    )

    mask_prob = torch.sigmoid(mask_logits) # quanto la query copre il pixel
    class_prob = torch.softmax(class_logits, dim=-1)[..., :-1] # probabilità che la query appartenga ad una classe

    pixel_scores = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
    pixel_scores = pixel_scores.squeeze(0)

    return pixel_scores
    

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
    parser.add_argument("--eomtName", default="local_drive_model")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    # liste vuote dove verranno salvati i punteggi anomalia
    anomaly_score_msp_list_EoMT = []
    anomaly_score_maxlogit_list_EoMT = []
    anomaly_score_maxentropy_list_EoMT = []
    anomaly_score_rba_list_EoMT = []
    ood_gts_list = [] # maschere ground truth OoD

    results_path = os.path.join(os.path.dirname(__file__), 'results.txt')
    file = open(results_path, 'w')
    
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # carica il modello
    model_EoMT = load_eomt(args, device)
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
    # ciclo su tutte le immagini
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        # images = images.permute(0,3,1,2)
        with torch.no_grad():

            # EoMT inference
            mask_logits_per_layer, class_logits_per_layer = model_EoMT(images)

            logits_EoMT = eomt_to_pixel_logits(
                mask_logits_per_layer,
                class_logits_per_layer
            )
        
            
        # anomaly scores
        scores_EoMT = anomaly_scores(logits_EoMT, use_rba=True)

        # ground truth OOD
        ood_gts = load_ood_gt(path)

        # salta immagini senza pixel OOD
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        
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
        del mask_logits_per_layer
        del class_logits_per_layer
        del scores_EoMT
        del ood_gts

        if device.type == "cuda":
            torch.cuda.empty_cache()
            
    file.write( "\n")
    
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
    
    # stampa EoMT
    print(f"AUPRC msp score EoMT: {prc_auc_msp_EoMT * 100.0}")
    print(f"FPR@TPR95 msp EoMT: {fpr_msp_EoMT * 100.0}")

    print(f"AUPRC maxlogit score EoMT: {prc_auc_maxlogit_EoMT * 100.0}")
    print(f"FPR@TPR95 maxlogit EoMT: {fpr_maxlogit_EoMT * 100.0}")

    print(f"AUPRC maxentropy score EoMT: {prc_auc_maxentropy_EoMT * 100.0}")
    print(f"FPR@TPR95 maxentropy EoMT: {fpr_maxentropy_EoMT * 100.0}")

    print(f"AUPRC rba score EoMT: {prc_auc_rba_EoMT * 100.0}")
    print(f"FPR@TPR95 rba EoMT: {fpr_rba_EoMT * 100.0}")
    
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
        
    
