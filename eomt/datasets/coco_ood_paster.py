import random
import cv2
import numpy as np
from pycocotools.coco import COCO

class CocoOODPaster:
    def __init__(
        self,
        coco_root,
        split="val2017",
        categories=None,
        min_area=500,
        max_tries=20,
        target_height_range=(80, 250),
    ):
        self.coco_root = coco_root
        self.split = split
        self.img_dir = f"{coco_root}/{split}"
        self.ann_file = f"{coco_root}/annotations/instances_{split}.json"

        self.coco = COCO(self.ann_file)

        if categories is None:
            categories = [
                "elephant", "giraffe", "zebra", "bear",
                "couch", "chair", "toaster", "microwave",
                "banana", "apple", "backpack"
            ]
        # oggetti che verranno incollati

        self.categories = categories
        self.cat_ids = self.coco.getCatIds(catNms=categories)

        print("cat_ids:", self.cat_ids)

        self.img_ids = []
        for cat_id in self.cat_ids:
            self.img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))

        self.img_ids = list(set(self.img_ids))

        if len(self.img_ids) == 0:
            raise ValueError(f"Nessuna immagine trovata per le categorie: {categories}")

        print("num img_ids:", len(self.img_ids))


        # immagini COCO che contengono almeno uno di quegli oggetti

        self.min_area = min_area
        self.max_tries = max_tries
        self.target_height_range = target_height_range

    def get_random_object(self):
        for _ in range(self.max_tries):
            img_id = random.choice(self.img_ids) # immagine casuale
            img_info = self.coco.loadImgs(img_id)[0] # info immagine

            ann_ids = self.coco.getAnnIds(
                imgIds=img_info["id"],
                catIds=self.cat_ids,
                iscrowd=False
            ) # categorie desiderate

            anns = self.coco.loadAnns(ann_ids)
            if len(anns) == 0:
                continue

            ann = random.choice(anns) # oggetto casuale

            if ann["area"] < self.min_area: # scarta oggetti piccoli
                continue

            img_path = f"{self.img_dir}/{img_info['file_name']}"
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converte RGB
            mask = self.coco.annToMask(ann).astype(np.uint8) # crea maschera (HxW)

            ys, xs = np.where(mask > 0) # pixel oggetto
            if len(xs) == 0 or len(ys) == 0:
                continue

            # bounding rectangle minimo
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            obj_img = img[ymin:ymax + 1, xmin:xmax + 1] # immagine oggetto
            obj_mask = mask[ymin:ymax + 1, xmin:xmax + 1] # maschera ritagliata

            if obj_img.shape[0] < 5 or obj_img.shape[1] < 5:
                continue

            cat_name = self.coco.loadCats([ann["category_id"]])[0]["name"]

            return obj_img, obj_mask, cat_name

        raise RuntimeError("Could not sample a valid COCO object.")

    def resize_object(self, obj_img, obj_mask):
        h, w = obj_img.shape[:2]

        target_h = random.randint(*self.target_height_range) # altezza casuale
        scale = target_h / h
        target_w = max(1, int(w * scale))

        obj_img = cv2.resize(obj_img, (target_w, target_h)) # resize immagine
        obj_mask = cv2.resize(
            obj_mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        ) # resize maschera

        return obj_img, obj_mask

    def paste(self, city_img):
        """
        city_img: RGB image, numpy array H x W x 3

        returns:
            city_paste: RGB image with pasted OOD object
            ood_mask: binary mask H x W, 1 on pasted object
        """

        city_paste = city_img.copy() # copia immagine
        H, W = city_paste.shape[:2]

        obj_img, obj_mask, cat_name = self.get_random_object()
        print("Oggetto incollato:", cat_name)
        obj_img, obj_mask = self.resize_object(obj_img, obj_mask) # resize

        h, w = obj_img.shape[:2]

        if h >= H or w >= W:
            obj_img = cv2.resize(obj_img, (W // 4, H // 4))
            obj_mask = cv2.resize(
                obj_mask,
                (W // 4, H // 4),
                interpolation=cv2.INTER_NEAREST
            )
            h, w = obj_img.shape[:2]

        # posizione casuale
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)

        roi = city_paste[y:y+h, x:x+w] # region of interest

        mask_bool = obj_mask > 0 # maschera booleana
        roi[mask_bool] = obj_img[mask_bool] # paste

        city_paste[y:y+h, x:x+w] = roi

        # maschera finale OoD
        ood_mask = np.zeros((H, W), dtype=np.uint8)
        ood_mask[y:y+h, x:x+w] = obj_mask
        print("DEBUG RETURN 3 VALUES")

        return city_paste, ood_mask, cat_name
