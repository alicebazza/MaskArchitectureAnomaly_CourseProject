# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions import *

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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObstacle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
    '--loadDir',
    default='/content/MaskArchitectureAnomaly_CourseProject/trained_models'
    )
    parser.add_argument("--erfnetWeights", default="erfnet_pretrained.pth")
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
    
    # carica il modello
    model_ERFNet = load_erfnet(args, device).to(device)

    
    for idx, path in enumerate(glob.glob(os.path.expanduser(str(args.input[0])))):
    # ciclo su tutte le immagini
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        # images = images.permute(0,3,1,2)
        with torch.no_grad():
            # ERFNet inference
            result_ERFNet = model_ERFNet(images)
            result_ERFNet = result_ERFNet[:, :-1, :, :] # togliamo no object
            logits_ERFNet = result_ERFNet.squeeze(0)
            
        # anomaly scores
        scores_ERFNet = anomaly_scores(logits_ERFNet, use_rba=False)
    
        # ground truth OOD
        ood_gts = load_ood_gt(path, size=(512, 1024))

        # salta immagini senza pixel OOD
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        
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

        if device.type == "cuda":
            torch.cuda.empty_cache()


    file.write( "\n")
    
    print("Numero immagini valide:", len(ood_gts_list))
    print("MSP scores:", len(anomaly_score_msp_list_ERFNet))

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
    
    del ood_gts_list
    del anomaly_score_msp_list_ERFNet
    del anomaly_score_maxlogit_list_ERFNet
    del anomaly_score_maxentropy_list_ERFNet
    del model_ERFNet
    
    file.close() # scriviamo su result.txt

if __name__ == '__main__':
    main()
