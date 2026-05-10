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
    
# per una classificazione con K classi, la confuzion_matrix è una matrice KxK
# righe -> classi reali, colonne -> classi predette
# elemento (i,j) -> numero di esempi nella classe reale i predetti come j
def compute_miou(confusion_matrix):
    intersection = np.diag(confusion_matrix) # true positive per classe

    gt_pixels = confusion_matrix.sum(axis=1)
    # numero totale di elementi veri per ciascuna classe
    pred_pixels = confusion_matrix.sum(axis=0)
    # numero totale di elementi predetti per ciascuna classe

    union = gt_pixels + pred_pixels - intersection

    iou = intersection / np.maximum(union, 1)
    miou = np.mean(iou)

    return miou, iou
    
def update_confusion_matrix(confusion_matrix, gt, pred, num_classes):
    valid = gt != 255

    gt_valid = gt[valid]
    pred_valid = pred[valid]

    valid_classes = (gt_valid >= 0) & (gt_valid < num_classes) & \
                    (pred_valid >= 0) & (pred_valid < num_classes)

    gt_valid = gt_valid[valid_classes]
    pred_valid = pred_valid[valid_classes]

    indices = num_classes * gt_valid + pred_valid
    cm = np.bincount(indices, minlength=num_classes ** 2)
    cm = cm.reshape(num_classes, num_classes)

    confusion_matrix += cm
    
def evaluate_cityscapes_miou(model, args, device):
    num_classes = NUM_CLASSES

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    image_paths = sorted(glob.glob(
        os.path.join(
            args.datadir,
            "leftImg8bit",
            "val",
            "*",
            "*_leftImg8bit.png"
        )
    ))

    print("Number of Cityscapes val images:", len(image_paths))

    if len(image_paths) == 0:
        raise RuntimeError("No Cityscapes validation images found. Check --datadir.")
        
    for idx, img_path in enumerate(image_paths):
        print(f"[{idx + 1}/{len(image_paths)}] {img_path}")

        gt_path = img_path.replace("leftImg8bit", "gtFine")
        gt_path = gt_path.replace("_leftImg8bit.png", "_gtFine_trainIds.png")

        if not os.path.exists(gt_path):
            raise RuntimeError(f"Ground truth not found: {gt_path}")

        image = Image.open(img_path).convert("RGB")
        gt = np.array(Image.open(gt_path))

        x = input_transform(image).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(x)

        logits = logits.squeeze(0)
        pred = torch.argmax(logits, dim=0).cpu().numpy()

        if pred.shape != gt.shape:
            pred = cv2.resize(
                pred.astype(np.uint8),
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        update_confusion_matrix(confusion_matrix, gt, pred, num_classes)

        del x, logits, pred, gt

        if device.type == "cuda":
            torch.cuda.empty_cache()

    miou, iou = compute_miou(confusion_matrix)
    print("\n==============================")
    print("Cityscapes validation results")
    print("==============================")
    print(f"mIoU: {miou * 100.0:.2f}")

    print("\nPer-class IoU:")
    for c, class_iou in enumerate(iou):
        print(f"Class {c}: {class_iou * 100.0:.2f}")

    results_path = os.path.join(os.path.dirname(__file__), "results_cityscapes_miou.txt")

    with open(results_path, "w") as f:
        f.write("Cityscapes validation results\n")
        f.write(f"mIoU: {miou * 100.0:.2f}\n\n")
        f.write("Per-class IoU:\n")

        for c, class_iou in enumerate(iou):
            f.write(f"Class {c}: {class_iou * 100.0:.2f}\n")

    print(f"\nResults saved to: {results_path}")




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
    
    model = load_erfnet(args, device)
    evaluate_cityscapes_miou(model, args, device)


if __name__ == '__main__':
    main()
