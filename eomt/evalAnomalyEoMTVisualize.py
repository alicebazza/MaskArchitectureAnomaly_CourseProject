"""
Visualizzazione ed evaluation di anomaly segmentation con EoMT.

Versione minimale e commentata.

Lo script fa quattro cose:
1. esegue EoMT sulle immagini in input;
2. salva visualizzazioni della predizione semantica e della ground truth OOD;
3. salva visualizzazioni delle mappe di anomaly score;
4. calcola metriche OOD globali e metriche OOD condizionate alla classe semantica predetta.

Richiede un file functions.py che definisca:
- load_eomt
- eomt_to_pixel_logits
- load_ood_gt
- anomaly_scores
- eval_score
"""

import csv
import glob
import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, ToTensor

try:
    from lightning import seed_everything
except ImportError:
    seed_everything = None

from functions import (
    anomaly_scores,
    eomt_to_pixel_logits,
    eval_score,
    load_eomt,
    load_ood_gt,
)


IGNORE_INDEX = 255
IMAGE_SIZE = (1024, 1024)
SCORE_NAMES = ["msp", "maxlogit", "entropy", "rba"]

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

CITYSCAPES_PALETTE = np.array(
    [
        [128, 64, 128], [244, 35, 232], [70, 70, 70],
        [102, 102, 156], [190, 153, 153], [153, 153, 153],
        [250, 170, 30], [220, 220, 0], [107, 142, 35],
        [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
    ],
    dtype=np.uint8,
)

OOD_PALETTE = {
    0: np.array([0.0, 0.0, 0.0], dtype=np.float32),
    1: np.array([1.0, 1.0, 1.0], dtype=np.float32),
    IGNORE_INDEX: np.array([0.0, 0.0, 0.0], dtype=np.float32),
}

INPUT_TRANSFORM = Compose([Resize(IMAGE_SIZE, Image.BILINEAR), ToTensor()])


def resolve_device(device_argument):
    """
    Decide se usare CPU o GPU.

    Se l'utente specifica esplicitamente --device, viene rispettata quella scelta.
    Altrimenti lo script usa CUDA quando disponibile, e CPU in caso contrario.
    """
    if device_argument is not None:
        return device_argument
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_eomt_model(config_path, state_dict_path, device_argument=None):
    """
    Carica configurazione, pesi e modello EoMT.

    La funzione imposta anche un seed fisso, così da rendere più riproducibili
    eventuali componenti stocastiche. Il modello viene costruito dalla funzione
    esterna load_eomt, definita in functions.py.

    Restituisce:
        model: modello EoMT pronto per inferenza;
        device: stringa "cpu" o "cuda".
    """
    if seed_everything is not None:
        seed_everything(0, verbose=False)
    else:
        torch.manual_seed(0)
        np.random.seed(0)

    device = resolve_device(device_argument)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    model = load_eomt(device=device, config=config, state_dict_path=state_dict_path)
    return model, device


def load_image(image_path, device):
    """
    Carica una immagine RGB e la converte nel formato atteso da EoMT.

    La trasformazione standardizza la dimensione a IMAGE_SIZE e produce un tensore
    C x H x W. In questo codice EoMT riceve valori uint8 in scala [0, 255], non
    float normalizzati in [0, 1].
    """
    original_image = Image.open(image_path).convert("RGB")
    image_tensor = INPUT_TRANSFORM(original_image).float()
    image_tensor = (image_tensor * 255).to(torch.uint8)
    return image_tensor.to(device)


def compute_pixel_logits(image_path, model, device):
    """
    Esegue l'inferenza EoMT su una singola immagine.

    I logits pixel-wise sono l'output principale del modello: vengono usati sia
    per ottenere la predizione semantica Cityscapes, sia per calcolare gli anomaly
    score. La funzione disattiva il calcolo dei gradienti perché siamo in fase di
    evaluation, non di training.
    """
    image_tensor = load_image(image_path, device)
    with torch.no_grad():
        pixel_logits = eomt_to_pixel_logits(image_tensor, device, model)
    return image_tensor, pixel_logits


def compute_semantic_prediction(pixel_logits):
    """
    Converte i logits del modello in una maschera semantica.

    Applica la softmax lungo la dimensione delle classi e poi prende l'argmax.
    Il risultato è una immagine H x W in cui ogni pixel contiene l'indice della
    classe Cityscapes predetta, da 0 a 18.
    """
    probabilities = F.softmax(pixel_logits.detach().cpu(), dim=0)
    return torch.argmax(probabilities, dim=0).numpy().astype(np.uint8)


def compute_anomaly_score_maps(pixel_logits):
    """
    Calcola le mappe di anomaly score a partire dai logits semantici.

    Gli score usati sono:
        msp: 1 - massima probabilità softmax;
        maxlogit: opposto del massimo logit;
        entropy: entropia della distribuzione softmax;
        rba: score RBA calcolato dalla funzione anomaly_scores.

    Restituisce un dizionario score_name -> mappa H x W.
    """
    msp, maxlogit, entropy, rba = anomaly_scores(pixel_logits.detach().cpu(), use_rba=True)
    return {
        "msp": msp.detach().cpu().numpy(),
        "maxlogit": maxlogit.detach().cpu().numpy(),
        "entropy": entropy.detach().cpu().numpy(),
        "rba": rba.detach().cpu().numpy(),
    }


def load_ood_ground_truth(image_path):
    """
    Carica la ground truth binaria OOD associata a una immagine.

    La convenzione attesa è:
        0: pixel in-distribution;
        1: pixel OOD/anomalo;
        255: pixel da ignorare nella valutazione.
    """
    return load_ood_gt(image_path, size=IMAGE_SIZE)


def cityscapes_mapping():
    """
    Crea il mapping classe Cityscapes -> colore RGB.

    Serve solo per visualizzare la predizione semantica in modo leggibile.
    I colori vengono normalizzati in [0, 1], come richiesto da matplotlib.
    """
    return {
        class_id: CITYSCAPES_PALETTE[class_id].astype(np.float32) / 255.0
        for class_id in range(len(CITYSCAPES_CLASSES))
    }


def apply_colormap(mask, mapping):
    """
    Trasforma una maschera di interi in una immagine RGB.

    Ogni valore della maschera viene interpretato come una classe e sostituito
    con il colore corrispondente nel dizionario mapping. Classi non presenti nel
    mapping vengono visualizzate in nero.
    """
    colored = np.zeros((*mask.shape, 3), dtype=np.float32)
    for class_id in np.unique(mask):
        colored[mask == class_id] = mapping.get(int(class_id), [0.0, 0.0, 0.0])
    return colored


def tensor_to_numpy_image(image_tensor):
    """
    Converte un tensore immagine PyTorch in un array NumPy visualizzabile.

    L'input ha forma C x H x W; l'output ha forma H x W x C. Se i valori sono
    in scala [0, 255], vengono riportati in [0, 1].
    """
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    return np.clip(image_np, 0.0, 1.0)


def plot_prediction_vs_gt(image_tensor, prediction, ood_gt, save_path):
    """
    Salva una figura con immagine, predizione semantica e ground truth OOD.

    Questa visualizzazione serve per controllare qualitativamente se la classe
    semantica predetta da EoMT è ragionevole e dove si trovano i pixel anomali
    secondo la ground truth.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tensor_to_numpy_image(image_tensor))
    axes[0].set_title("Image")
    axes[1].imshow(apply_colormap(prediction, cityscapes_mapping()))
    axes[1].set_title("EoMT semantic prediction")
    axes[2].imshow(apply_colormap(ood_gt, OOD_PALETTE))
    axes[2].set_title("OOD ground truth")

    for ax in axes:
        ax.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_anomaly_scores(image_tensor, score_maps, save_path):
    """
    Salva una figura con immagine originale e mappe di anomaly score.

    Le mappe mostrano dove ciascun criterio assegna un valore alto di anomalia.
    Sono utili per confrontare qualitativamente MSP, MaxLogit, Entropy e RBA
    sulla stessa immagine.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(tensor_to_numpy_image(image_tensor))
    axes[0].set_title("Image")
    axes[0].axis("off")

    for ax, score_name in zip(axes[1:], SCORE_NAMES):
        im = ax.imshow(score_maps[score_name])
        ax.set_title(score_name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def create_metric_storage():
    """
    Inizializza gli accumulatori per le metriche OOD.

    Mantiene due livelli di valutazione:
        global: metriche calcolate su tutti i pixel validi del dataset;
        by_predicted_class: metriche calcolate separatamente sui pixel che EoMT
        ha assegnato a ciascuna classe Cityscapes.

    La seconda parte serve a capire se il detector OOD funziona meglio o peggio
    quando il modello interpreta i pixel come road, car, person, building, ecc.
    """
    return {
        "global": {
            "ood_gts": [],
            "scores": {score_name: [] for score_name in SCORE_NAMES},
        },
        "by_predicted_class": {
            class_id: {
                "ood_gts": [],
                "scores": {score_name: [] for score_name in SCORE_NAMES},
                "valid_pixels": 0,
                "ood_pixels": 0,
                "id_pixels": 0,
            }
            for class_id in range(len(CITYSCAPES_CLASSES))
        },
    }


def add_metrics_for_image(ood_gt, prediction, score_maps, metric_storage):
    """
    Aggiunge una immagine agli accumulatori delle metriche.

    Per le metriche globali, vengono raccolti tutti i pixel validi della immagine.
    Per le metriche per classe, i pixel vengono prima separati in base alla classe
    semantica predetta da EoMT, poi per ciascuna classe si raccolgono GT OOD/ID e
    anomaly score.

    Una classe contribuisce alle metriche solo se contiene sia pixel OOD sia pixel
    ID; altrimenti AUPRC e FPR@TPR95 non sono ben definiti per quella classe.
    """
    valid_mask = ood_gt != IGNORE_INDEX
    if not np.any(valid_mask):
        return

    binary_gt = (ood_gt == 1).astype(np.uint8)

    if np.any(binary_gt[valid_mask] == 1):
        metric_storage["global"]["ood_gts"].append(binary_gt[valid_mask])
        for score_name in SCORE_NAMES:
            metric_storage["global"]["scores"][score_name].append(score_maps[score_name][valid_mask])

    for class_id in range(len(CITYSCAPES_CLASSES)):
        class_mask = valid_mask & (prediction == class_id)
        if not np.any(class_mask):
            continue

        class_gt = binary_gt[class_mask]
        class_storage = metric_storage["by_predicted_class"][class_id]
        class_storage["valid_pixels"] += int(class_mask.sum())
        class_storage["ood_pixels"] += int(class_gt.sum())
        class_storage["id_pixels"] += int(class_mask.sum() - class_gt.sum())

        if np.any(class_gt == 1) and np.any(class_gt == 0):
            class_storage["ood_gts"].append(class_gt)
            for score_name in SCORE_NAMES:
                class_storage["scores"][score_name].append(score_maps[score_name][class_mask])


def safe_eval_score(ood_gts, scores):
    """
    Calcola AUPRC e FPR@TPR95 gestendo i casi non valutabili.

    eval_score può fallire se mancano esempi positivi o negativi. In quel caso
    la funzione restituisce (None, None), così il resto dello script può continuare
    e lasciare vuote le metriche non definite.
    """
    if len(ood_gts) == 0:
        return None, None
    try:
        return eval_score(ood_gts, scores)
    except ValueError:
        return None, None


def print_global_metrics(metric_storage):
    """
    Stampa le metriche OOD globali per ogni anomaly score.

    Queste sono le metriche standard sull'intero dataset, senza distinguere per
    classe semantica predetta.
    """
    print("\nMetriche globali OOD")
    for score_name in SCORE_NAMES:
        auprc, fpr = safe_eval_score(
            metric_storage["global"]["ood_gts"],
            metric_storage["global"]["scores"][score_name],
        )
        if auprc is None:
            print(f"{score_name}: non calcolabile")
        else:
            print(f"AUPRC {score_name}: {auprc * 100.0:.4f}")
            print(f"FPR@TPR95 {score_name}: {fpr * 100.0:.4f}")


def save_class_metrics_csv(metric_storage, output_dir):
    """
    Salva su CSV le metriche OOD condizionate alla classe predetta.

    Ogni riga identifica una coppia:
        classe Cityscapes predetta, anomaly score.

    Le colonne principali sono:
        valid_pixels: pixel validi predetti come quella classe;
        ood_pixels: tra questi, pixel OOD secondo la ground truth;
        id_pixels: tra questi, pixel in-distribution;
        auprc: AUPRC per distinguere OOD da ID dentro quella classe;
        fpr_at_tpr95: FPR al 95% di TPR dentro quella classe.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ood_metrics_by_predicted_class.csv"

    fieldnames = [
        "class_id", "class_name", "valid_pixels", "ood_pixels", "id_pixels",
        "ood_pixel_percentage", "score", "auprc", "fpr_at_tpr95",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for class_id, class_storage in metric_storage["by_predicted_class"].items():
            valid_pixels = class_storage["valid_pixels"]
            if valid_pixels == 0:
                continue

            ood_percentage = 100.0 * class_storage["ood_pixels"] / valid_pixels
            for score_name in SCORE_NAMES:
                auprc, fpr = safe_eval_score(
                    class_storage["ood_gts"],
                    class_storage["scores"][score_name],
                )
                writer.writerow(
                    {
                        "class_id": class_id,
                        "class_name": CITYSCAPES_CLASSES[class_id],
                        "valid_pixels": valid_pixels,
                        "ood_pixels": class_storage["ood_pixels"],
                        "id_pixels": class_storage["id_pixels"],
                        "ood_pixel_percentage": ood_percentage,
                        "score": score_name,
                        "auprc": "" if auprc is None else auprc,
                        "fpr_at_tpr95": "" if fpr is None else fpr,
                    }
                )

    return csv_path


def print_class_metrics(metric_storage):
    """
    Stampa una sintesi leggibile delle metriche OOD per classe predetta.

    Vengono mostrate solo le classi per cui la valutazione è possibile, cioè le
    classi che hanno accumulato pixel sia OOD sia ID. Il CSV salvato da
    save_class_metrics_csv resta comunque il riferimento completo.
    """
    print("\nMetriche OOD per classe semantica predetta")
    for class_id, class_storage in metric_storage["by_predicted_class"].items():
        if class_storage["valid_pixels"] == 0:
            continue
        if len(class_storage["ood_gts"]) == 0:
            continue

        print(f"\n{CITYSCAPES_CLASSES[class_id]}:")
        print(
            f"  pixel validi: {class_storage['valid_pixels']} | "
            f"OOD: {class_storage['ood_pixels']} | ID: {class_storage['id_pixels']}"
        )
        for score_name in SCORE_NAMES:
            auprc, fpr = safe_eval_score(
                class_storage["ood_gts"],
                class_storage["scores"][score_name],
            )
            if auprc is not None:
                print(f"  {score_name:<8} AUPRC={auprc * 100.0:7.4f}  FPR@TPR95={fpr * 100.0:7.4f}")


def process_image(image_path, model, device, output_dir, metric_storage, save_scores=True):
    """
    Processa una singola immagine dall'inizio alla fine.

    Per ogni immagine esegue:
        1. inferenza EoMT;
        2. predizione semantica;
        3. caricamento ground truth OOD;
        4. calcolo anomaly score;
        5. salvataggio visualizzazione prediction-vs-GT;
        6. eventuale salvataggio delle mappe di score;
        7. aggiornamento delle metriche globali e per classe.
    """
    image_tensor, pixel_logits = compute_pixel_logits(image_path, model, device)
    prediction = compute_semantic_prediction(pixel_logits)
    ood_gt = load_ood_ground_truth(image_path)
    score_maps = compute_anomaly_score_maps(pixel_logits)

    output_dir = Path(output_dir)
    image_stem = Path(image_path).stem

    prediction_path = output_dir / f"{image_stem}_prediction_vs_gt.png"
    plot_prediction_vs_gt(image_tensor, prediction, ood_gt, prediction_path)

    score_path = None
    if save_scores:
        score_path = output_dir / f"{image_stem}_anomaly_scores.png"
        plot_anomaly_scores(image_tensor, score_maps, score_path)

    add_metrics_for_image(ood_gt, prediction, score_maps, metric_storage)

    del pixel_logits
    if device == "cuda":
        torch.cuda.empty_cache()

    return prediction_path, score_path


def collect_image_paths(input_pattern):
    """
    Restituisce la lista di immagini da processare.

    L'input può essere un path singolo oppure un glob pattern, per esempio:
        /data/RoadAnomaly/images/*.jpg

    La lista viene ordinata per rendere l'esecuzione riproducibile.
    """
    expanded = os.path.expanduser(str(input_pattern))
    if os.path.isfile(expanded):
        return [expanded]
    return sorted(glob.glob(expanded))


def main():
    """
    Entry point dello script da riga di comando.

    Carica il modello, raccoglie le immagini, processa il dataset e infine stampa
    o salva le metriche. Le opzioni principali permettono di scegliere input,
    cartella di output, configurazione, pesi del modello, device e salvataggio
    delle mappe di anomaly score.
    """
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, help="Path singolo o glob delle immagini da processare.")
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/ml_anomaly_segmentation/visualizations",
        help="Cartella in cui salvare visualizzazioni e CSV.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
        help="Path della config EoMT.",
    )
    parser.add_argument(
        "--state-dict-path",
        default="/content/drive/MyDrive/ml_anomaly_segmentation/eomt_cityscapes.bin",
        help="Path del file .bin con i pesi del modello.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device da usare. Se omesso, usa CUDA quando disponibile.",
    )
    parser.add_argument(
        "--no-score-plots",
        action="store_true",
        help="Se presente, non salva le mappe anomaly.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Se presente, non calcola metriche globali e per classe.",
    )
    args = parser.parse_args()

    model, device = load_eomt_model(args.config_path, args.state_dict_path, args.device)

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata con input: {args.input}")

    metric_storage = create_metric_storage()

    for image_path in image_paths:
        print(f"Processo: {image_path}")
        prediction_path, score_path = process_image(
            image_path=image_path,
            model=model,
            device=device,
            output_dir=args.output_dir,
            metric_storage=metric_storage,
            save_scores=not args.no_score_plots,
        )
        print(f"  Prediction vs GT salvata in: {prediction_path}")
        if score_path is not None:
            print(f"  Anomaly scores salvati in: {score_path}")

    if not args.no_metrics:
        print_global_metrics(metric_storage)
        print_class_metrics(metric_storage)
        csv_path = save_class_metrics_csv(metric_storage, args.output_dir)
        print(f"\nMetriche per classe salvate in: {csv_path}")


if __name__ == "__main__":
    main()

