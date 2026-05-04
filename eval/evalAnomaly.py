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
    anomaly_score_msp_list_ERFNet = []
    anomaly_score_maxlogit_list_ERFNet = []
    anomaly_score_maxentropy_list_ERFNet = []
    anomaly_score_msp_list_EoMT = []
    anomaly_score_maxlogit_list_EoMT = []
    anomaly_score_maxentropy_list_EoMT = []
    anomaly_score_rba_list_EoMT = []
    
    ood_gts_list = [] # maschere ground truth OOD

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    device = torch.device("cpu" if args.cpu else "cuda")
    model_ERFNet = ERFNet(NUM_CLASSES).to(device)

    if (not args.cpu):
        model_ERFNet = torch.nn.DataParallel(model)
    
    encoder = ViT(
    img_size=(512, 1024),
    patch_size=14,
    backbone_name="vit_large_patch14_reg4_dinov2",
    )

    model_EoMT = EoMT(
    encoder=encoder,
    num_classes=NUM_CLASSES,
    num_q=100,
    num_blocks=4,
    masked_attn_enabled=True,
    )

    checkpoint = torch.load(weightspath, map_location=device)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]

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

    model_ERFNet = load_my_state_dict(model_ERFNet, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model_ERFNet.eval()
    
    model_EoMT.load_state_dict(checkpoint, strict=False)

    model_EoMT = model_EoMT.to(device)
    model_EoMT.eval()
    
    def anomaly(logits, model):
        anomaly_result = []
    
        probs = torch.softmax(logits, dim=0)
        anomaly_result_msp = 1.0 - torch.max(probs, dim=0)[0]
        anomaly_result.append(anomaly_result_msp)

        anomaly_result_maxlogit = -torch.max(logits, dim=0)[0]
        anomaly_result.append(anomaly_result_maxlogit)

        K = probs.shape[0]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=0)
        entropy_normalized = entropy / torch.log(torch.tensor(float(K), device=probs.device))
        anomaly_result_maxentropy = entropy_normalized
        anomaly_result.append(anomaly_result_maxentropy)
        
        if model == model_EoMT
            anomaly_result_rba = -torch.tanh(logits).sum(dim=0)
            anomaly_result.append(anomaly_result_rba)
        
    return [anomaly_result]
        
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
    # ciclo su tutte le immagini
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        # images = images.permute(0,3,1,2) probabilmente sbagliato
        with torch.no_grad():
            result = model_ERFNet(images)
            mask_logits_per_layer, class_logits_per_layer = model_EoMT(images)
        
            
        logits_ERFNet = result.squeeze(0)   # [C, H, W]
        
        mask_logits = mask_logits_per_layer[-1]        # [B, Q, Hm, Wm]
        class_logits = class_logits_per_layer[-1]      # [B, Q, C+1]

        # porta le mask alla risoluzione dell'immagine/GT
        mask_logits = torch.nn.functional.interpolate(
            mask_logits,
            size=(512, 1024),
            mode="bilinear",
            align_corners=False,
        )
        
        mask_prob = torch.sigmoid(mask_logits)              # [B, Q, H, W]
        class_prob = torch.softmax(class_logits, dim=-1)    # [B, Q, C+1]

        # togli classe no-object
        class_prob = class_prob[:, :, :-1]                  # [B, Q, C]
        
        pixel_probs = torch.einsum("bqc, bqhw -> bchw", class_prob, mask_prob)
        pixel_probs = pixel_probs / (pixel_probs.sum(dim=1, keepdim=True) + 1e-8)

        probs = pixel_probs.squeeze(0)      # [C, H, W]
        logits_EoMT = torch.log(probs + 1e-8)    # pseudo-logits
        
        anomaly_result_ERFNet = anomaly(logits_ERFNet, model_ERFNet)
        anomaly_result_EoMT = anomaly(logits_EoMT, model_EoMT)
    

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
                anomaly_score_msp_list_ERFNet.append(anomaly_result_ERFNet[0].cpu().numpy())
                anomaly_score_maxlogit_list_ERFNet.append(anomaly_result_ERFNet[1].cpu().numpy())
                anomaly_score_maxentropy_list_ERFNet.append(anomaly_result_ERFNet[2].cpu().numpy())
                anomaly_score_msp_list_EoMT.append(anomaly_result_EoMT[0].cpu().numpy())
                anomaly_score_maxlogit_list_EoMT.append(anomaly_result_EoMT[1].cpu().numpy())
                anomaly_score_maxentropy_list_EoMT.append(anomaly_result_EoMT[2].cpu().numpy())
                anomaly_score_rba_list_EoMT.append(anomaly_result_EoMT[3].cpu().numpy())
        del result, anomaly_result_ERFNet, anomaly_result_EoMT, ood_gts, mask
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
    
    [prc_auc_msp_ERFNet, fpr_msp_ERFNet] = eval_score(ood_gts_list, anomaly_score_msp_list_ERFNEt)
    [prc_auc_maxlogit_ERFNet, fpr_maxlogit_ERFNet] = eval_score(ood_gts_list, anomaly_score_maxlogit_list_ERFNet)
    [prc_auc_maxentropy_ERFNet, fpr_maxentropy_ERFNet] = eval_score(ood_gts_list, anomaly_score_maxentropy_list_ERFNet)
    [prc_auc_msp_EoMt, fpr_msp_EoMT] = eval_score(ood_gts_list, anomaly_score_msp_list_EoMT)
    [prc_auc_maxlogit_EoMT, fpr_maxlogit_EoMT] = eval_score(ood_gts_list, anomaly_score_maxlogit_list_EoMT)
    [prc_auc_maxentropy_EoMT, fpr_maxentropy_EoMT] = eval_score(ood_gts_list, anomaly_score_maxentropy_list_EoMT)
    [prc_auc_rba_EoMT, fpr_rba_EoMT] = eval_score(ood_gts_list, anomaly_score_rba_list_EoMT)
    
    print(f'AUPRC msp score ERFNet: {prc_auc_msp_ERFNet*100.0}')
    print(f'FPR@TPR95 msp ERFNet: {fpr_msp_ERFNEt*100.0}')

    print(f'AUPRC maxlogit score ERFNet: {prc_auc_maxlogit_ERFNet*100.0}')
    print(f'FPR@TPR95 maxlogit ERFNet: {fpr_maxlogit_ERFNet*100.0}')

    print(f'AUPRC maxentropy score ERFNet: {prc_auc_maxentropy_ERFNet*100.0}')
    print(f'FPR@TPR95 maxentropy ERFNet: {fpr_maxentropy_ERFNet*100.0}')

    file.write(('    AUPRC msp score ERFNet:' + str(prc_auc_msp_ERFNet*100.0) + '   FPR@TPR95 msp ERFNet:' + str(fpr_msp_ERFNet*100.0) +
                '\n    AUPRC maxlogit score ERFNet:' + str(prc_auc_maxlogit_ERFNet*100.0) + '   FPR@TPR95 maxlogit ERFNet:' + str(fpr_maxlogit_ERFNet*100.0) +
                '\n    AUPRC maxentropy score ERFNet:' + str(prc_auc_maxentropy_ERFNet*100.0) + '   FPR@TPR95 maxentropy ERFNet:' + str(fpr_maxentropy_ERFNet*100.0)))
    
    print(f'AUPRC msp score EoMT: {prc_auc_msp_EoMT*100.0}')
    print(f'FPR@TPR95 msp EoMT: {fpr_msp_EoMT*100.0}')

    print(f'AUPRC maxlogit score EoMT: {prc_auc_maxlogit_EoMT*100.0}')
    print(f'FPR@TPR95 maxlogit EoMT: {fpr_maxlogit_EoMT*100.0}')

    print(f'AUPRC maxentropy score EoMT: {prc_auc_maxentropy_EoMT*100.0}')
    print(f'FPR@TPR95 maxentropy EoMT: {fpr_maxentropy_EoMT*100.0}')
        
    print(f'AUPRC rba score EoMT: {prc_auc_rba_EoMT*100.0}')
    print(f'FPR@TPR95 rba EoMT: {fpr_rba_EoMT*100.0}')

    file.write(('    AUPRC msp score EoMT:' + str(prc_auc_msp_EoMT*100.0) + '   FPR@TPR95 msp EoMT:' + str(fpr_msp_EoMT*100.0) +
                '\n    AUPRC maxlogit score EoMT:' + str(prc_auc_maxlogit_EoMT*100.0) + '   FPR@TPR95 maxlogit EoMT:' + str(fpr_maxlogit_EoMT*100.0) +
                '\n    AUPRC maxentropy score EoMT:' + str(prc_auc_maxentropy_EoMT*100.0) + '   FPR@TPR95 maxentropy EoMT:' + str(fpr_maxentropy_EoMT*100.0))+
                '\n    AUPRC rba score EoMT:' + str(prc_auc_rba_EoMT*100.0) + '   FPR@TPR95 rba EoMT:' + str(fpr_rba_EoMT*100.0))
    
    file.close() # scriviamo su result.txt

if __name__ == '__main__':
    main()
