import random
import cv2
import numpy as np
from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image

class CocoOODPaster:

    """
    Classe che estrae un oggetto da COCO e lo incolla su un'immagine RGB.
    L'oggetto incollato viene trattato come oggetto OoD.

    Input principali:
        coco_root: cartella principale del dataset COCO
        split: sottoinsieme COCO, per esempio "val2017"
        categories: categorie COCO da cui estrarre gli oggetti
        target_height_range: intervallo di altezze per l'oggetto incollato
        num_fixed_images: numero massimo di immagini COCO considerate.

    Output:
        un oggetto CocoOODPaster pronto per incollare oggetti OoD.
    """
    
    def __init__(
        self,
        coco_root: str | Path,
        split="val2017",
        categories=None,
        target_height_range=(80, 250),
        num_fixed_images=300,
    ):
        self.coco_root = Path(coco_root)
        self.split = split
        self.img_dir = self.coco_root / split
        self.ann_file = self.coco_root / "annotations" / f"instances_{split}.json"
        # costruisce il path delle immagini e quello delle annotazioni
        self.target_height_range = target_height_range

        if categories is None:
            categories = [
                "elephant", "giraffe", "zebra", "bear",
                "toaster", "microwave",
                "banana", "apple", "backpack"
            ]

        self.categories = categories
        self.coco = COCO(str(self.ann_file))
        self.cat_ids = self.coco.getCatIds(catNms=list(categories))
        # converte i nomi delle categorie negli ID COCO

        # Prendiamo solo immagini che contengono almeno una categoria desiderata
        img_ids: list[int] = []
        for cat_id in self.cat_ids:
            img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))

        self.img_ids = list(dict.fromkeys(img_ids))
        random.shuffle(self.img_ids)
        self.img_ids = self.img_ids[:num_fixed_images]
        # rimuove i duplicati mantenendo l'ordine
        # e prende num_fixed_images casuali
        if not self.img_ids:
            raise ValueError("Nessuna immagine COCO trovata per le categorie OOD.")

    def get_random_object(self, max_tries: int=20):
        """
        Estrae casualmente un oggetto valido da COCO.

        Input:
            max_tries: numero massimo di tentativi di estrazione.

        Output:
            obj_img: immagine RGB ritagliata dell'oggetto
            obj_mask: maschera binaria dell'oggetto ritagliato
            cat_name: nome della categoria COCO dell'oggetto.

        """
        
        for _ in range(max_tries):
    
            img_id = random.choice(self.img_ids)
            # sceglie casualmente un'immagine tra quelle selezionate (300)
            img_info = self.coco.loadImgs(img_id)[0]
            # carica le informazioni dell'immagine scelta

            ann_ids = self.coco.getAnnIds(
                imgIds=img_id,
                catIds=self.cat_ids,
                iscrowd=False
            ) # ID delle annotazioni dell'immagine scelta
            
            # ogni annotazione corrisponde ad un oggetto nell'immagine
            # contiene: ID dell'immagine a cui appartiene, categoria, bounding box

            anns = self.coco.loadAnns(ann_ids)
            if not anns:
                continue
            # carica le annotazioni, se non ce ne sono continua
                
            ann = random.choice(anns)
            # sceglie un'annotazione casuale (oggetto da ritagliare)

            mask = self.coco.annToMask(ann).astype(np.uint8)
            # converte l'annotazione in una maschera binaria (1 oggetto, 0 fuori)
            if not mask.any():
                continue

            ys, xs = np.where(mask > 0)
            # trova i pixel appartenenti all'oggetto
            
            # bounding box dell'oggetto
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()
            
            # apre l'immagine COCO
            img_path = self.img_dir / img_info["file_name"]
            img = np.asarray(Image.open(img_path).convert("RGB"))

            obj_img = img[ymin:ymax + 1, xmin:xmax + 1]
            # ritaglia dall'immagine originale la regione contenente l'oggetto
            obj_mask = mask[ymin:ymax + 1, xmin:xmax + 1]
            # ritaglia allo stesso modo la maschera
            
            # hanno dimensione rettangolare ma poi in paste viene
            # effettivamente incollato solo l'oggeto tramite una maschera

            cat_name = self.coco.loadCats([ann["category_id"]])[0]["name"]
            # categoria associata all'annotazione scelta

            return obj_img, obj_mask, cat_name
            
        raise RuntimeError("Impossibile estrarre un oggetto COCO valido.")

    def resize_object(self, obj_img, obj_mask):
        """
        Ridimensiona un oggetto mantenendo le proporzioni.

        Input:
            obj_img: immagine RGB dell'oggetto, array h x w x 3
            obj_mask: maschera binaria dell'oggetto, array h x w.

        Output:
            obj_img: immagine dell'oggetto ridimensionata
            obj_mask: maschera ridimensionata.

        """
        h, w = obj_img.shape[:2] # dimensione oggetto

        target_h = random.randint(*self.target_height_range)
        # sceglie casualmente una nuova altezza nel range
        target_w = max(1, round(w * target_h / h))
        # nuova larghezza mantenendo le proporzioni

        obj_img = cv2.resize(
            obj_img,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )
        # ridimensiona l'immagine dell'oggetto
        
        obj_mask = cv2.resize(
            obj_mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        ) # ridimensiona la maschera

        return obj_img, obj_mask


    def paste(self, city_img):
        """
        Incolla un oggetto COCO su un'immagine RGB Cityscapes.

        Input:
            city_img: immagine RGB di destinazione, array H x W x 3.

        Output:
            city_paste: immagine RGB con l'oggetto OoD incollato
            ood_mask: maschera binaria H x W dell'oggetto incollato
            cat_name: nome della categoria COCO incollata.

        """

        city_paste = city_img.copy()
        H, W = city_paste.shape[:2]

        obj_img, obj_mask, cat_name = self.get_random_object()
        # estrae casualmente un oggetto da COCO
        obj_img, obj_mask = self.resize_object(obj_img, obj_mask)
        # ridimensiona casualmemte l'oggetto

        h, w = obj_img.shape[:2]
        
        # Se l'oggetto è più grande dell'immagine Cityscapes, lo ridimensiona
        if w > W or h > H:
            scale = min(W / w, H / h) * 0.8

            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            obj_mask = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            h, w = obj_img.shape[:2]
                
        # sceglie casualmente le coordinate in cui incollare l'oggetto
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)

        roi = city_paste[y:y+h, x:x+w]
        # regione dell'immagine in cui verrà incollato l'oggetto

        mask_bool = obj_mask > 0 # maschera oggetto
        roi[mask_bool] = obj_img[mask_bool]
        # copia nella roi (regione di interesse) solo i pixel dell'oggetto

        city_paste[y:y+h, x:x+w] = roi
        # mette la roi nell'immagine completa

        ood_mask = np.zeros((H, W), dtype=np.uint8)
        ood_mask[y:y+h, x:x+w] = obj_mask # maschera oggetto incollato

        return city_paste, ood_mask, cat_name
