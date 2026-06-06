# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union, Optional, Sequence
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from eomt.datasets.lightning_data_module import LightningDataModule
from eomt.datasets.dataset import Dataset
from eomt.datasets.transforms import Transforms
from eomt.datasets.coco_ood_paster import CocoOODPaster
from eomt.datasets.ood_wrapper import OODDatasetWrapper
from eomt.datasets.cityscapes_semantic import CityscapesSemantic


class CityscapesSemanticOE(CityscapesSemantic):
    """
    DataModule Cityscapes con Outlier Exposure da COCO.

    Input:
        path:
            percorso del dataset Cityscapes
        coco_root:
            percorso del dataset COCO
        p_ood:
            probabilità con cui un'immagine di training viene modificata
            incollando un oggetto OOD da COCO
        coco_split:
            split COCO da usare, per esempio "val2017" o "train2017"
        ood_categories:
            lista opzionale di categorie COCO da usare come oggetti OOD
        ood_target_height_range:
            range di altezza, in pixel, degli oggetti OOD incollati
        **kwargs:
            altri argomenti passati a CityscapesSemantic.

    Output:
        self.cityscapes_train_dataset:
            Training set Cityscapes wrappato con OODDatasetWrapper.
        self.cityscapes_val_dataset:
            Validation set Cityscapes standard, senza OOD.

    Cosa fa:
        costruisce normalmente i dataset Cityscapes tramite CityscapesSemantic.
        Durante il setup di training, sostituisce il training set con un wrapper
        che, con probabilità p_ood, incolla oggetti presi da COCO sulle immagini
        Cityscapes. La validation resta invariata.
    """

    def __init__(
        self,
        path,
        coco_root: str | Path,
        p_ood: float = 0.1,
        coco_split: str = "val2017", # quale porzione di coco usare
        ood_categories: Optional[Sequence[str]] = None, # filtrare quali categorie di coco utilizzare
        ood_target_height_range: tuple[int, int] = (80, 250), # altezza minima e massima che deve avere l'oggetto
        **kwargs,
    ) -> None:
        super().__init__(path=path, **kwargs)
        # estensione della classe cityscapes_semantic

        if not 0.0 <= p_ood <= 1.0:
            raise ValueError(f"p_ood must be in [0, 1], got {p_ood}")

        self.coco_root = Path(coco_root)
        self.p_ood = p_ood
        self.coco_split = coco_split
        self.ood_categories = ood_categories
        self.ood_target_height_range = ood_target_height_range

    def setup(self, stage=None):
        super().setup(stage)
            # costruisce normalmente training e validation set

        if stage in (None, "fit"): # solo durante l'addestramento
            paster = CocoOODPaster(
                coco_root=self.coco_root,
                split=self.coco_split,
                categories=self.ood_categories,
                target_height_range=self.ood_target_height_range,
            )

            if not isinstance(self.cityscapes_train_dataset, OODDatasetWrapper):
            # controlla che il training dataset non sia già stato wrappato
                self.cityscapes_train_dataset = OODDatasetWrapper(
                    base_dataset=self.cityscapes_train_dataset,
                    paster=paster,
                    p_ood=self.p_ood,
                )

        return self
