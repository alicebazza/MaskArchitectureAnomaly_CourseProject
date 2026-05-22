import random
import cv2
import numpy as np
from pycocotools.coco import COCO

# prende un oggetto dal dataset COCO e lo incolla dentro un'immagine
# viene trattato come oggetto OoD
class CocoOODPaster:
    def __init__(
        self,
        coco_root, # cartella del dataset COCO
        split="val2017", # sottoinsieme del dataset da usare
        categories=None, # categorie di COCO da cui prendere oggetti
        target_height_range=(80, 250), # intervallo per l'altezza dell'oggetto incollato
    ):
        self.coco_root = coco_root
        self.split = split
        self.img_dir = f"{coco_root}/{split}" # percorso cartella con immagini
        self.ann_file = f"{coco_root}/annotations/instances_{split}.json" # percorso cartella con annotazioni

        self.coco = COCO(self.ann_file) # carica le annotazioni usando pycocotools

        if categories is None:
            categories = [
                "elephant", "giraffe", "zebra", "bear",
                "couch", "chair", "toaster", "microwave",
                "banana", "apple", "backpack"
            ]

        self.categories = categories
        self.cat_ids = self.coco.getCatIds(catNms=categories)
        # converte i nomi delle categorie negli ID COCO

        self.img_ids = []
        # lista che contiene gli ID delle immagini COCO contenenti almeno una categoria scelta
        for cat_id in self.cat_ids:
            self.img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))

        self.img_ids = list(set(self.img_ids)) # rimuove duplicati

        if len(self.img_ids) == 0:
            raise ValueError("Nessuna immagine trovata per le categorie scelte.")

        self.target_height_range = target_height_range

    def get_random_object(self):
    # metodo che estrae casualmente un oggetto dal dataset COCO
    
        img_id = random.choice(self.img_ids)
        # sceglie casualmente un'immagine tra quelle selezionate
        img_info = self.coco.loadImgs(img_id)[0]
        # carica le informazioni dell'immagine scelta

        ann_ids = self.coco.getAnnIds(
            imgIds=img_id,
            catIds=self.cat_ids,
            iscrowd=False
        ) # ID delle annotazioni dell'immagine scelta
        
        # ogni annotazione corrisponde ad un oggetto nell'immagine
        # contiene: ID dell'immagine a cui appartiene, categoria, bounding box

        ann = random.choice(self.coco.loadAnns(ann_ids))
        # sceglie un'annotazione casuale (oggetto da ritagliare)

        img_path = f"{self.img_dir}/{img_info['file_name']}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self.coco.annToMask(ann).astype(np.uint8)
        # converte l'annotazione in una maschera binaria

        ys, xs = np.where(mask > 0)
        # trova i pixel appartenenti all'oggetto
        
        # bounding box dell'oggetto
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        obj_img = img[ymin:ymax + 1, xmin:xmax + 1]
        # ritaglia dall'immagine originale la regione contenente l'oggetto
        obj_mask = mask[ymin:ymax + 1, xmin:xmax + 1]
        # ritaglia allo stesso modo la maschera
        
        # hanno dimensione rettangolare ma poi in paste viene
        # effettivamente incollato solo l'oggeto tramite una maschera

        cat_name = self.coco.loadCats([ann["category_id"]])[0]["name"]
        # categoria associata all'annotazione scelta

        return obj_img, obj_mask, cat_name
        # restituisce immagine ritagliata, maschera e categoria

    def resize_object(self, obj_img, obj_mask):
    # metodo che ridimensiona l'oggetto mantenendo le proprozioni
        h, w = obj_img.shape[:2] # dimensione oggetto

        target_h = random.randint(*self.target_height_range)
        # sceglie casualmente una nuova altezza nel range
        scale = target_h / h
        target_w = int(w * scale)
        # nuova larghezza mantenendo le proporzioni

        obj_img = cv2.resize(obj_img, (target_w, target_h))
        # ridimensiona l'immagine dell'oggetto
        
        obj_mask = cv2.resize(
            obj_mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        ) # ridimensiona la maschera

        return obj_img, obj_mask
        # restituisce oggetto e maschera ridimensionate

    def paste(self, city_img):
        """
        city_img: immagine RGB, array numpy H x W x 3

        returns:
            city_paste: immagine RGB con oggetto OOD incollato
            ood_mask: maschera binaria H x W
            cat_name: nome della categoria incollata
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
        # copia nella roi solo i pixel dell'oggetto

        city_paste[y:y+h, x:x+w] = roi
        # mette la roi nell'immagine completa

        ood_mask = np.zeros((H, W), dtype=np.uint8)
        ood_mask[y:y+h, x:x+w] = obj_mask # maschera oggetto incollato

        return city_paste, ood_mask, cat_name
