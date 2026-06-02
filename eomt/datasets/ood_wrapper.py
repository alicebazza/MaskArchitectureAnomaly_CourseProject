import random
import numpy as np
import torch
from torchvision import tv_tensors
from coco_ood_paster import CocoOODPaster

# target è un dizionario
# target = {
#    "masks": tensor di forma [N, H, W], maschere binarie oggetti
#    "labels": tensor di forma [N], classe di ogni oggetto
#    "is_crowd": tensor di forma [N]
#    "ood_mask": maschera dei pixel OoD
#    "ood_category": categoria dell'oggetto OoD incollato

# N = numero di oggetti presenti nell'immagine



def _clone_target(target: dict) -> dict:
    """
    Input:
        target: dizionario contenente tensori, maschere, label e metadati.

    Output:
        cloned: nuovo dizionario con tensori clonati.
        
    Cosa fa:
        Duplica il dizionario target evitando modifiche in-place.
    """
    cloned = {}
    for key, value in target.items():
    # scorre tutte le coppie chiave-valore del dizionario
    # se il valore è un tensore Pytorch lo clona,
    # altrimenti lo resituisce così
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned
# se faccio solo .copy() faccio una shallow copy
    
    
class OODDatasetWrapper(torch.utils.data.Dataset):
    """
    Input:
        base_dataset: dataset che restituisce coppie (img, target)
        paster: oggetto con metodo paste che incolla un oggetto OoD
        p_ood: probabilità di applicare l'incollaggio OoD

    Output:
        img: immagine originale oppure modificata con oggetto OoD
        target: dizionario aggiornato con:
            - "ood_mask": maschera booleana [H, W] dei pixel OoD
            - "ood_category": categoria OoD incollata, oppure None

    Cosa fa:
        Wrappa un dataset esistente e, con probabilità p_ood, incolla un oggetto
        out-of-distribution sull'immagine, aggiornando il target con la maschera OoD.
    """
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        paster: CocoOODPaster,
        p_ood: float =0.1
        )-> None:
        self.base_dataset = base_dataset # dataset originale
        self.paster = paster # oggetto che sa incollare OoD su un'immagine
        self.p_ood = p_ood # probabilità di applicare il paste

    def __len__(self):
        """Restituisce la lunghezza del dataset base."""
        return len(self.base_dataset)
        # il wrapper ha la stessa lunghezza del dataset originale

    def __getitem__(self, idx):
        """
        Input:
            idx: indice dell'elemento da recuperare dal dataset.

        Output:
            img: immagine originale oppure modificata con un oggetto OoD.
            target: dizionario delle annotazioni aggiornato con:
                - maschere eventualmente corrette dopo l'occlusione OoD;
                - "ood_mask": maschera booleana dei pixel OoD;
                - "ood_category": categoria dell'oggetto OoD incollato
                  (oppure None se non è stato applicato alcun paste).

        Cosa fa:
            Recupera un campione dal dataset originale e, con probabilità p_ood,
            incolla un oggetto out-of-distribution sull'immagine.
        """
    
        img, target = self.base_dataset[idx]
        # immagini e annotazioni del dataset originale

        # evita di modificare in-place il target del dataset originale
        target = _clone_target(target)
        
        H, W = img.shape[-2:]
        
        if random.random() > self.p_ood:
            target["ood_mask"] = torch.zeros((H,W), dtype=torch.bool)
            # non viene applicato nessun oggetto OoD
            # restituisce una maschera tutta falsa
            target["ood_category"] =None
            # nessuna categoria OoD
            return img, target

        original_img = img
        original_target = _clone_target(target)
        # salva immagine e target originali nel caso in cui
        # il paste renda invisibili tutte le istanze originali

        # torch CHW -> numpy HWC uint8
        img_np = img.detach().cpu()
        img_np = img_np.permute(1, 2, 0).numpy()

        # controllo che l'immagine sia in [0,1]
        if img_np.max() <= 1.0:
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # incolla oggetto sull'immagine
        pasted_img_np, ood_mask_np, cat_name = self.paster.paste(img_np)
        # converte la maschera numpy in tensore booleano
        # i pixel con valore positivo diventano True
        ood_mask = torch.from_numpy(ood_mask_np > 0).bool()

        if ood_mask.shape != (H, W):
            raise ValueError(
                f"ood_mask ha forma {ood_mask.shape}, ma attesa {(H, W)}."
            )

        if "masks" in target and target["masks"].numel() > 0:
        # controlla che target contenga maschere e che non siano vuote
        
            visible_masks = target["masks"].bool() & ~ood_mask
            # rimuove dalle maschere i pixel coperti dall'oggetto OoD
            # se un oggetto viene coperto dal paste la sua maschera viene ridotta
            valid = visible_masks.flatten(1).any(dim=1)
            # controlla quali oggetti hanno ancora almeno un pixel visibile

            if not valid.any():
            # se non c'è alcun oggetto visibile
            # annulla le modifiche e restituisce l'immagine originale
                original_target["ood_mask"] = torch.zeros((H, W), dtype=torch.bool)
                original_target["ood_category"] = None
                return original_img, original_target

            target["masks"] = tv_tensors.Mask(visible_masks[valid])
            # tiene solo le maschere ancora valide

            if "labels" in target:
                target["labels"] = target["labels"][valid]
                # tiene le labe di oggetti visibili
            
            if "is_crowd" in target:
                target["is_crowd"] = target["is_crowd"][valid]

        target["ood_mask"] = ood_mask
        target["ood_category"] = cat_name

        # numpy --> PyTorch
        img_tensor = torch.from_numpy(pasted_img_np).permute(2, 0, 1).contiguous()

        # Preserva dtype e scala dell'immagine originale.
        if torch.is_floating_point(original_img):
            img_tensor = img_tensor.to(dtype=original_img.dtype) / 255.0
        else:
            img_tensor = img_tensor.to(dtype=original_img.dtype)

        return tv_tensors.Image(img_tensor), target

        
