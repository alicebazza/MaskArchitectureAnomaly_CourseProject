# Copyright (c) OpenMMLab. All rights reserved.
import os
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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    # liste vuote dove verranno salvati i punteggi anomalia
    anomaly_score_msp_list = []
    anomaly_score_maxlogit_list = []
    anomaly_score_maxentropy_list = []
    ood_gts_list = [] # maschere ground truth OOD

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    device = torch.device("cpu" if args.cpu else "cuda")
    model = ERFNet(NUM_CLASSES).to(device)

    if (not args.cpu):
        model = torch.nn.DataParallel(model)

    # carica i pesi del modello preaddestrato nel modello ERFNet che ho appena creato
    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
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

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
    # ciclo su tutte le immagini
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        # images = images.permute(0,3,1,2) probabilmente sbagliato
        with torch.no_grad():
            result = model(images)
        logits = result.squeeze(0)   # [C, H, W]

        probs = torch.softmax(logits, dim=0)
        anomaly_result_msp = 1.0 - torch.max(probs, dim=0)[0]

        anomaly_result_maxlogit = -torch.max(logits, dim=0)[0]

        K = probs.shape[0]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=0)
        entropy_normalized = entropy / torch.log(torch.tensor(float(K), device=probs.device))
        anomaly_result_maxentropy = entropy_normalized

        pathGT = path.replace("images", "labels_masks")
        # costruisce il path della maschera a partire dal path dell'immagine
        # cambia estensione a seconda del dataset
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT) # legge la maschera
        mask = target_transform(mask) # resize
        ood_gts = np.array(mask) # trasforma in array
        # maschera groud-truth dal dataset
            
        # ogni dataset ha codifiche diverse, le uniforma a:
        # 0 = in-distribution, 1 = OoD, 255 =ignore
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
        # se non c'è nessun pixel OoD salta immagine
            continue
        else:
                ood_gts_list.append(ood_gts)
                anomaly_score_msp_list.append(anomaly_result_msp.cpu().numpy())
                anomaly_score_maxlogit_list.append(anomaly_result_maxlogit.cpu().numpy())
                anomaly_score_maxentropy_list.append(anomaly_result_maxentropy.cpu().numpy())
        del result, anomaly_result_msp, anomaly_result_maxlogit,anomaly_result_maxentropy, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

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

        return [prc_auc, fpr]
    
    [prc_auc_msp, fpr_msp] = eval_score(ood_gts_list, anomaly_score_msp_list)
    [prc_auc_maxlogit, fpr_maxlogit] = eval_score(ood_gts_list, anomaly_score_maxlogit_list)
    [prc_auc_maxentropy, fpr_maxentropy] = eval_score(ood_gts_list, anomaly_score_maxentropy_list)

    print(f'AUPRC msp score: {prc_auc_msp*100.0}')
    print(f'FPR@TPR95 msp: {fpr_msp*100.0}')

    print(f'AUPRC maxlogit score: {prc_auc_maxlogit*100.0}')
    print(f'FPR@TPR95 maxlogit: {fpr_maxlogit*100.0}')

    print(f'AUPRC maxentropy score: {prc_auc_maxentropy*100.0}')
    print(f'FPR@TPR95 maxentropy: {fpr_maxentropy*100.0}')

    file.write(('    AUPRC msp score:' + str(prc_auc_msp*100.0) + '   FPR@TPR95 msp:' + str(fpr_msp*100.0) +
                '\n    AUPRC maxlogit score:' + str(prc_auc_maxlogit*100.0) + '   FPR@TPR95 maxlogit:' + str(fpr_maxlogit*100.0) +
                '\n    AUPRC maxentropy score:' + str(prc_auc_maxentropy*100.0) + '   FPR@TPR95 maxentropy:' + str(fpr_maxentropy*100.0)))
    file.close() # scriviamo su result.txt

if __name__ == '__main__':
    main()
