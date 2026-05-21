import random
import numpy as np
import torch
from torchvision import tv_tensors

# prende un dataset esistente
# prende ogni elemento di tale dataset e con una certa probabilità lo modifica
class OODDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, paster, p_ood=0.5):
        self.base_dataset = base_dataset # dataset originale
        self.paster = paster # oggetto che sa incollare OoD su un'immagine
        self.p_ood = p_ood # probabilità di applicare il paste

    def __len__(self):
        return len(self.base_dataset)
        # ha la stessa lunghezza del dataset originale

    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]
        # prende immagine e target dal dataset originale
        # target è un dizionario
        # target['mask'] = ha dim HxW e ad ogni pixel associa una classe da 0 a 18
        # target['ood_mask'] = maschera booeala con true -> pixel ID e false -> pixel OoD
        # target['ood_category'] = nome dell'oggetto OoD incollato

        if random.random() > self.p_ood:
            target["ood_mask"] = torch.zeros(img.shape[-2:], dtype=torch.bool)
            # non viene applicato nessun oggetto OoD
            # restituisce una maschera tutta falsa
            return img, target

        img_np = img.permute(1, 2, 0).cpu().numpy()

        # valori tra 0 e 255
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        pasted_img, ood_mask, cat_name = self.paster.paste(img_np)

        pasted_img = torch.from_numpy(pasted_img).permute(2, 0, 1)
        pasted_img = tv_tensors.Image(pasted_img)

        target["ood_mask"] = torch.from_numpy(ood_mask > 0)
        # maschera booleana: i pixel con valore >1 diventano true
        target["ood_category"] = cat_name
        # nome della categoria incollata
        
        return pasted_img, target
