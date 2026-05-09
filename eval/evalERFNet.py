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
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from evalAnomaly import *


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
    
    ood_gts_list = [] # maschere ground truth OoD

    results_path = os.path.join(os.path.dirname(__file__), 'results.txt')
    file = open(results_path, 'w')
    
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # carica il modello modelli
    model_ERFNet = load_erfnet(args, device)
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
    # ciclo su tutte le immagini
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        # images = images.permute(0,3,1,2)
        with torch.no_grad():
            # ERFNet inference
            result_ERFNet = model_ERFNet(images)
            logits_ERFNet = result_ERFNet.squeeze(0)
            
        # anomaly scores
        scores_ERFNet = anomaly_scores(logits_ERFNet, use_rba=False)

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

        
        del images
        del result_ERFNet
        del logits_ERFNet
        del scores_ERFNet
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
    
    
    # stampa ERFNet
    print(f"AUPRC msp score ERFNet: {prc_auc_msp_ERFNet * 100.0}")
    print(f"FPR@TPR95 msp ERFNet: {fpr_msp_ERFNet * 100.0}")

    print(f"AUPRC maxlogit score ERFNet: {prc_auc_maxlogit_ERFNet * 100.0}")
    print(f"FPR@TPR95 maxlogit ERFNet: {fpr_maxlogit_ERFNet * 100.0}")

    print(f"AUPRC maxentropy score ERFNet: {prc_auc_maxentropy_ERFNet * 100.0}")
    print(f"FPR@TPR95 maxentropy ERFNet: {fpr_maxentropy_ERFNet * 100.0}")
    
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
    
    file.close() # scriviamo su result.txt

if __name__ == '__main__':
    main()
