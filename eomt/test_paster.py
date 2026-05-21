import cv2
import matplotlib.pyplot as plt
from coco_ood_paster import CocoOODPaster


coco_root = "/content/datasets/coco"
city_img_path = "/content/datasets/cityscapes/example.png"

paster = CocoOODPaster(
    coco_root=coco_root,
    split="val2017"
)

city_img = cv2.imread(city_img_path)
city_img = cv2.cvtColor(city_img, cv2.COLOR_BGR2RGB)

img_ood, ood_mask = paster.paste(city_img)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Originale")
plt.imshow(city_img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Con oggetto OoD")
plt.imshow(img_ood)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Maschera OoD")
plt.imshow(ood_mask, cmap="gray")
plt.axis("off")

plt.show()
