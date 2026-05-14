# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # per guardare sia eval che eomt

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from eval.dataset import cityscapes
from eomt.models.eomt import EoMT
from eomt.models.vit import ViT
from eval.transform import Relabel, ToLabel, Colorize
from eval.iouEval import iouEval, getColorEntry
from evalEoMT import load_eomt
from eval.evalAnomaly import *

# configurazione e trasformazione dei dati
NUM_CHANNELS = 3
NUM_CLASSES = 20 # numero di categorie di oggetti che il modello può riconoscere

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((1024, 1024), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

target_transform_cityscapes = Compose([
    Resize((1024, 1024), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   # in Cityscapes le aree non classificate sono marcate con il valore 255 ---> rimappate alla classe 19
])

def main(args):

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")
    
    # carica il modello
    model_EoMT = load_eomt(args, device)

    model_EoMT.eval()

    dataset_val = cityscapes(
    args.datadir,
    input_transform_cityscapes,
    target_transform_cityscapes,
    subset=args.subset
    )

    loader = DataLoader(
        dataset_val,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    # fase di validazione e valutazione metrica
    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        labels = labels.long()
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        # non calcoliamo i gradienti
        with torch.no_grad():
            mask_logits_per_layer, class_logits_per_layer = model_EoMT(inputs)

            logits_EoMT = eomt_to_pixel_logits(
                mask_logits_per_layer,
                class_logits_per_layer
            )

        # scegliamo la classe con il punteggio più alto per ogni singolo pixel
        # confrontiamo la predizione del modello con la label
        prediction = logits_EoMT.max(0)[1].unsqueeze(0).unsqueeze(0)

        iouEvalVal.addBatch(prediction.data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        print (step, filenameSave)

    # calcolo precisione media e specifica
    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)
    
    """
    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")"
    print("=======================================")
    """
    
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--erfnetWeights', default="erfnet_pretrained.pth")
    parser.add_argument("--eomtName", default="local_drive_model")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/content/drive/MyDrive/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
