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
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.1,]
    )
    args = parser.parse_args()

    # salviamo i logits per non far ricalcolare tutto alla GPU ogni volta
    logits_dir = "saved_logits_erfnet"
    # nome della cartella dove salviamo i logits
    os.makedirs(logits_dir, exist_ok=True)
    # crea la cartella se non esiste già
    
    temperatures = args.temperatures
    # dizionario per contenere i punteggi di anomalia per ogni temperatura
    anomaly_score_msp_temp_ERFNet = {T: [] for T in temperatures}
    
    ood_gts_list = [] # maschere ground truth OoD

    results_path = os.path.join(os.path.dirname(__file__), 'results.txt')
    file = open(results_path, 'w')
    
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # carica il modello se non siamo in modalita eval
    model_ERFNet = None
    if not args.eval_only:
        model_ERFNet = load_erfnet(args, device)
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
    # ciclo su tutte le immagini
        print(path)
        
        ood_gts = load_ood_gt(path)

        # salta immagini senza pixel OoD
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)

        img_name = os.path.basename(path)
        img_name = os.path.splitext(img_name)[0] + ".pt"
        logits_path = os.path.join(logits_dir, img_name)
        # costruisce il percorso completo del file dei logit salvati

        if os.path.exists(logits_path):
            logits_ERFNet = torch.load(logits_path, map_location="cpu")
            # se esistono già li carica
        else:
            images = input_transform(
                Image.open(path).convert('RGB')).unsqueeze(0).float().to(device)
                
            with torch.no_grad():
                result_ERFNet = result_ERFNet[:, :-1, :, :]
                logits_ERFNet = result_ERFNet.squeeze(0).cpu()

            torch.save(logits_ERFNet, logits_path)

            del images
            del result_ERFNet

        logits_ERFNet = logits_ERFNet.to(device)
        
        # temperature scaling
        for T in temperatures:
            logits_temp = logits_ERFNet / T
            scores_temp = anomaly_scores(logits_temp, use_rba=False)

            # MSP anomaly score con temperatura
            anomaly_score_msp_temp_ERFNet[T].append(
                scores_temp[0].detach().cpu().numpy()
            )

            del logits_temp
            del scores_temp

        del logits_ERFNet
        del ood_gts
        
        if device.type == "cuda":
            torch.cuda.empty_cache()

    file.write("\nERFNet temperature scaling\n")
    
    best_T_auprc = None
    best_auprc = -1.0
    best_fpr_at_best_auprc = None

    best_T_fpr = None
    best_fpr = float("inf")
    best_auprc_at_best_fpr = None

    for T in temperatures:
        prc_auc_msp, fpr_msp = eval_score(
            ood_gts_list,
            anomaly_score_msp_temp_ERFNet[T]
        )

        print(f"T={T}: AUPRC MSP ERFNet: {prc_auc_msp * 100.0}")
        print(f"T={T}: FPR@TPR95 MSP ERFNet: {fpr_msp * 100.0}")

        file.write(
            f"T={T}: AUPRC MSP ERFNet: {prc_auc_msp * 100.0} "
            f"FPR@TPR95 MSP ERFNet: {fpr_msp * 100.0}\n"
        )

        # migliore secondo AUPRC: più alto è meglio
        if prc_auc_msp > best_auprc:
            best_auprc = prc_auc_msp
            best_fpr_at_best_auprc = fpr_msp
            best_T_auprc = T

        # migliore secondo FPR95: più basso è meglio
        if fpr_msp < best_fpr:
            best_fpr = fpr_msp
            best_auprc_at_best_fpr = prc_auc_msp
            best_T_fpr = T
    
    print("\nBest temperature according to AUPRC")
    print(f"Best T AUPRC: {best_T_auprc}")
    print(f"Best AUPRC: {best_auprc * 100.0}")
    print(f"Corresponding FPR@TPR95: {best_fpr_at_best_auprc * 100.0}")

    print("\nBest temperature according to FPR@TPR95")
    print(f"Best T FPR95: {best_T_fpr}")
    print(f"Best FPR@TPR95: {best_fpr * 100.0}")
    print(f"Corresponding AUPRC: {best_auprc_at_best_fpr * 100.0}")

    file.write(
        "\nBest temperature according to AUPRC\n"
        f"Best T AUPRC: {best_T_auprc}\n"
        f"Best AUPRC: {best_auprc * 100.0}\n"
        f"Corresponding FPR@TPR95: {best_fpr_at_best_auprc * 100.0}\n\n"
    )

    file.write(
        "Best temperature according to FPR@TPR95\n"
        f"Best T FPR95: {best_T_fpr}\n"
        f"Best FPR@TPR95: {best_fpr * 100.0}\n"
        f"Corresponding AUPRC: {best_auprc_at_best_fpr * 100.0}\n\n"
    )

    file.close()


if __name__ == '__main__':
    main()
