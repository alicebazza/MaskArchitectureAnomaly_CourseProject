import random
import numpy as np
import torch
from torchvision import tv_tensors


class OODDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, paster, p_ood=0.5):
        self.base_dataset = base_dataset
        self.paster = paster
        self.p_ood = p_ood

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]

        if random.random() > self.p_ood:
            target["ood_mask"] = torch.zeros(img.shape[-2:], dtype=torch.bool)
            return img, target

        img_np = img.permute(1, 2, 0).cpu().numpy()

        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        pasted_img, ood_mask, cat_name = self.paster.paste(img_np)

        pasted_img = torch.from_numpy(pasted_img).permute(2, 0, 1)
        pasted_img = tv_tensors.Image(pasted_img)

        target["ood_mask"] = torch.from_numpy(ood_mask > 0)
        target["ood_category"] = cat_name

        return pasted_img, target
