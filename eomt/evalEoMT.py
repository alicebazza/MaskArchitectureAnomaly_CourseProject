# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
import torch
import warnings
import yaml

from PIL import Image
from torch.nn import functional as F
import numpy as np
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode

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

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObstacle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
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
    
    config_path = 'configs/dinov2/cityscapes/semantic/eomt_base_640.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    state_dict_path = '/content/drive/MyDrive/eomt_cityscapes.bin'
    
    warnings.filterwarnings("ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )
    
    # carica il modello
    model_EoMT = load_eomt(device, config, state_dict_path)
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
    # ciclo su tutte le immagini
        print(path)
        image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        # images = images.permute(0,3,1,2)
        with torch.no_grad():
            image = image.squeeze(0)
            image = (image * 255).to(torch.uint8)
            logits_EoMT = eomt_to_pixel_logits(image, device, model_EoMT)

            
        # anomaly scores
        scores_EoMT = anomaly_scores(logits_EoMT, use_rba=True)

        # ground truth OOD
        ood_gts = load_ood_gt(path, size=(1024, 1024))

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
        
    
