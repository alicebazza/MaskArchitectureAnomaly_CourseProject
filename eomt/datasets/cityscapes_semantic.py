# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms
from datasets.coco_ood_paster import CocoOODPaster
from datasets.ood_wrapper import OODDatasetWrapper


class CityscapesSemantic(LightningDataModule):
    def __init__(
        self,
        path, # cartella dove stanno gli zip cityscapes
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        # crea le trasformazioni da applicare al dataset di training
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, **kwargs):
    # questo metodo trasforma una singola immagine con labelId
    # in una lista di maschere separate + train_id
        masks, labels = [], []

        for label_id in target[0].unique():
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            if cls is None or cls.ignore_in_eval:
                continue

            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        # mask = lista di maschere binarie
        # labels = lista delle classi di training
        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.path, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.path, "gtFine_trainvaltest.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        # crea il dataset di training
        self.cityscapes_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            **cityscapes_dataset_kwargs,
        )
        # crea il dataset di validation
        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            **cityscapes_dataset_kwargs,
        )
        # nessuna trasformazione

        return self

    # crea il dataloader di training
    def train_dataloader(self):
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    # crea il dataloader di validation
    def val_dataloader(self):
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

class CityscapesSemanticOE(CityscapesSemantic):
    def __init__(
        self,
        path, # percorso cityscapes
        coco_root, # percorso coco
        p_ood=0.5,
        **kwargs
    ):
        super().__init__(
            path=path,
            num_classes=19,
            **kwargs
        )
        # costruisce un classico cityscapes con 19 classi

        self.coco_root = coco_root
        self.p_ood = p_ood

    def setup(self, stage=None):
        super().setup(stage)
        # costruisce normalmente training e validation set

        paster = CocoOODPaster(
            coco_root=self.coco_root,
            split="val2017",
            target_height_range=(80, 250), # TODO da valutare se aumentare
        )
        # Crea un oggetto che prende istanze da COCO val2017 e le incolla sulle immagini Cityscapes.

        self.cityscapes_train_dataset = OODDatasetWrapper(
            base_dataset=self.cityscapes_train_dataset,
            paster=paster,
            p_ood=self.p_ood,
            ood_label=self.ood_label,
        )
        # sostituisce il training dataset con un wrapper

        return self
