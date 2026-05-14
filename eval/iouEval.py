# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch

class iouEval:

    def __init__(self, nClasses, ignoreIndex=19):
        # la classe 19 viene ignorata nei calcoli
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1
        self.reset()

    # creazione di vettori inizializzati con degli zeri per salvare i punteggi
    def reset (self):
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()        

    def addBatch(self, x, y):
        # x = predizioni del modello, y = etichette reali (ground truth)
        # input ---> [Batch, Classi, Altezza, Larghezza]

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        # one hot encoding
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            # riempie il tensore di zeri mettendo un "1" nella posizione corrispondente alla classe predetta
            x_onehot.scatter_(1, x, 1).float()
        else:
            # se è già in formato one-hot lo converte in numeri decimali
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        # gestione dell'indice da ignorare
        if (self.ignoreIndex != -1):
            ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            # rimuove la classe da ignorare da entrambi i tensori
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores=0

            #print(type(x_onehot))
            #print(type(y_onehot))
            #print(x_onehot.size())
            #print(y_onehot.size())

        tpmult = x_onehot * y_onehot    # true positive
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores) # false positive
        # sommiamo i pixel per ottenere il totale dei falsi positivi per ogni classe
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot) # false negative
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp  # numeratore: i pixel correttamente identificati
        # denominatore: unione di TP, FP e FN.
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den # calcola il punteggio per ogni singola classe

        # ritorna: 1. La media di tutte le classi (mIoU), 2. Il vettore con i punteggi singoli
        return torch.mean(iou), iou
    
# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

