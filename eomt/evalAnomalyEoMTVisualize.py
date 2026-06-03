"""
Visualizzazione ed evaluazione anomaly segmentation con EoMT.

Questo script usa le funzioni definite in functions.py:
- load_eomt
- eomt_to_pixel_logits
- load_ood_gt
- anomaly_scores
- eval_score

Esempio:
python evalAnomalyEoMTVisualize_rewritten.py \
    --input "/content/dataset/RoadAnomaly/images/*.jpg" \
    --output-dir "/content/drive/MyDrive/ml_anomaly_segmentation/visualizations" \
    --config-path "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" \
    --state-dict-path "/content/drive/MyDrive/ml_anomaly_segmentation/eomt_cityscapes.bin"
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

CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CITYSCAPES_PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)

OOD_PALETTE = {
    0: np.array([0.0, 0.0, 0.0], dtype=np.float32),
    1: np.array([1.0, 1.0, 1.0], dtype=np.float32),
    IGNORE_INDEX: np.array([0.0, 0.0, 0.0], dtype=np.float32),
}

INPUT_TRANSFORM = Compose(
    [
        Resize(IMAGE_SIZE, Image.BILINEAR),
        ToTensor(),
    ]
)


def resolve_device(device_argument):
    """Restituisce il device richiesto, oppure cuda se disponibile."""
    if device_argument is not None:
        return device_argument
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_eomt_model(config_path, state_dict_path, device_argument=None):
    """Carica configurazione, pesi e modello EoMT."""
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
    """Carica una immagine RGB e la converte nel tensore atteso da EoMT."""
    original_image = Image.open(image_path).convert("RGB")
    image_tensor = INPUT_TRANSFORM(original_image).float()

    # EoMT in questo codice lavora su tensori uint8 in scala [0, 255].
    image_tensor = (image_tensor * 255).to(torch.uint8)
    return image_tensor.to(device)


def compute_pixel_logits(image_path, model, device):
    """Esegue inferenza su una immagine e restituisce immagine e logits pixel-wise."""
    image_tensor = load_image(image_path, device)

    with torch.no_grad():
        pixel_logits = eomt_to_pixel_logits(image_tensor, device, model)

    return image_tensor, pixel_logits


def compute_semantic_prediction(pixel_logits):
    """Calcola la maschera semantica predetta, con classi Cityscapes 0..18."""
    probabilities = F.softmax(pixel_logits.detach().cpu(), dim=0)
    prediction = torch.argmax(probabilities, dim=0).numpy().astype(np.uint8)
    return prediction


def cityscapes_mapping(ignore_index=IGNORE_INDEX):
    """Crea mapping fisso classe Cityscapes -> colore RGB in scala [0, 1]."""
    mapping = {
        class_id: CITYSCAPES_PALETTE[class_id].astype(np.float32) / 255.0
        for class_id in range(len(CITYSCAPES_CLASSES))
    }
    mapping[ignore_index] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return mapping


def apply_colormap(mask, mapping):
    """Trasforma una maschera di ID in una immagine RGB."""
    colored = np.zeros((*mask.shape, 3), dtype=np.float32)

    for class_id in np.unique(mask):
        colored[mask == class_id] = mapping.get(int(class_id), [0.0, 0.0, 0.0])

    return colored


def tensor_to_numpy_image(image_tensor):
    """Converte un tensore CxHxW in array HxWxC in scala [0, 1]."""
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    return np.clip(image_np, 0.0, 1.0)


def plot_prediction_vs_gt(image_tensor, prediction, ood_gt, save_path):
    """Salva una figura con immagine, predizione semantica e ground truth OOD."""
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


def tensor_score_to_numpy(score_tensor):
    """Converte una anomaly map PyTorch in array NumPy."""
    return score_tensor.detach().cpu().numpy()


def compute_anomaly_score_maps(pixel_logits):
    """
    Calcola anomaly scores.

    1. MSP = 1 - max softmax probability
    2. MaxLogit = - max logit
    3. Entropy normalizzata
    4. RBA, se use_rba=True
    """
    msp, maxlogit, entropy, rba = anomaly_scores(pixel_logits.detach().cpu(), use_rba=True)

    return {
        "msp": tensor_score_to_numpy(msp),
        "maxlogit": tensor_score_to_numpy(maxlogit),
        "entropy": tensor_score_to_numpy(entropy),
        "rba": tensor_score_to_numpy(rba),
    }


def plot_anomaly_scores(image_tensor, score_maps, save_path):
    """Salva una figura con immagine e mappe di anomaly score."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(tensor_to_numpy_image(image_tensor))
    axes[0].set_title("Image")
    axes[0].axis("off")

    for ax, (score_name, score_map) in zip(axes[1:], score_maps.items()):
        im = ax.imshow(score_map)
        ax.set_title(score_name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def load_ood_ground_truth(image_path):
    """Carica la ground truth OOD."""
    return load_ood_gt(image_path, size=IMAGE_SIZE)


def create_metric_storage():
    """Inizializza gli accumulatori per metriche."""
    return {
        "ood_gts": [],
        "scores": {
            "msp": [],
            "maxlogit": [],
            "entropy": [],
            "rba": [],
        },
    }


def add_metrics_for_image(ood_gt, score_maps, metric_storage):
    """Aggiunge GT e score di una immagine agli accumulatori."""
    if 1 not in np.unique(ood_gt):
        print("  Metriche saltate: la ground truth non contiene anomalie.")
        return

    metric_storage["ood_gts"].append(ood_gt)

    for score_name, score_map in score_maps.items():
        metric_storage["scores"][score_name].append(score_map)


def print_metric_results(metric_storage):
    """Calcola e stampa AUPRC e FPR@TPR95 per ogni anomaly score."""
    if len(metric_storage["ood_gts"]) == 0:
        print("Metriche non calcolate: nessuna immagine contiene anomalie nella ground truth.")
        return

    for score_name in ["msp", "maxlogit", "entropy", "rba"]:
        auprc, fpr = eval_score(
            metric_storage["ood_gts"],
            metric_storage["scores"][score_name],
        )
        print(f"AUPRC {score_name}: {auprc * 100.0:.4f}")
        print(f"FPR@TPR95 {score_name}: {fpr * 100.0:.4f}")


def create_prediction_stats():
    """Inizializza accumulatori per capire quali classi Cityscapes vengono predette."""
    num_classes = len(CITYSCAPES_CLASSES)
    return {
        "all_pixels": np.zeros(num_classes, dtype=np.int64),
        "ood_pixels": np.zeros(num_classes, dtype=np.int64),
        "ind_pixels": np.zeros(num_classes, dtype=np.int64),
        "num_images": 0,
        "num_images_with_ood": 0,
        "per_image_rows": [],
    }


def class_count_rows(counts, total=None):
    """Converte un vettore di conteggi per classe in righe leggibili/salvabili."""
    if total is None:
        total = int(counts.sum())

    rows = []
    for class_id, count in enumerate(counts):
        count = int(count)
        percentage = 0.0 if total == 0 else 100.0 * count / total
        rows.append(
            {
                "class_id": class_id,
                "class_name": CITYSCAPES_CLASSES[class_id],
                "pixels": count,
                "percentage": percentage,
            }
        )
    rows.sort(key=lambda row: row["pixels"], reverse=True)
    return rows


def update_prediction_stats(image_path, prediction, ood_gt, prediction_stats):
    """
    Aggiorna le statistiche sulle classi predette.

    all_pixels: distribuzione delle classi predette su tutti i pixel validi.
    ood_pixels: distribuzione delle classi predette solo dove la GT OOD vale 1.
    ind_pixels: distribuzione delle classi predette solo dove la GT OOD vale 0.
    """
    valid_mask = ood_gt != IGNORE_INDEX
    ood_mask = ood_gt == 1
    ind_mask = ood_gt == 0

    prediction_stats["num_images"] += 1
    if np.any(ood_mask):
        prediction_stats["num_images_with_ood"] += 1

    all_counts = np.bincount(
        prediction[valid_mask].ravel(), minlength=len(CITYSCAPES_CLASSES)
    )[: len(CITYSCAPES_CLASSES)]
    ood_counts = np.bincount(
        prediction[ood_mask].ravel(), minlength=len(CITYSCAPES_CLASSES)
    )[: len(CITYSCAPES_CLASSES)]
    ind_counts = np.bincount(
        prediction[ind_mask].ravel(), minlength=len(CITYSCAPES_CLASSES)
    )[: len(CITYSCAPES_CLASSES)]

    prediction_stats["all_pixels"] += all_counts
    prediction_stats["ood_pixels"] += ood_counts
    prediction_stats["ind_pixels"] += ind_counts

    ood_total = int(ood_counts.sum())
    if ood_total > 0:
        dominant_ood_class_id = int(np.argmax(ood_counts))
        dominant_ood_class_name = CITYSCAPES_CLASSES[dominant_ood_class_id]
        dominant_ood_percentage = 100.0 * int(ood_counts[dominant_ood_class_id]) / ood_total
    else:
        dominant_ood_class_id = -1
        dominant_ood_class_name = "none"
        dominant_ood_percentage = 0.0

    prediction_stats["per_image_rows"].append(
        {
            "image": str(image_path),
            "valid_pixels": int(all_counts.sum()),
            "ood_pixels": ood_total,
            "dominant_ood_predicted_class_id": dominant_ood_class_id,
            "dominant_ood_predicted_class_name": dominant_ood_class_name,
            "dominant_ood_predicted_percentage": dominant_ood_percentage,
        }
    )


def save_prediction_stats_csv(prediction_stats, output_dir):
    """Salva CSV con statistiche aggregate e per-immagine sulle predizioni."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = output_dir / "prediction_class_statistics.csv"
    with open(aggregate_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["region", "class_id", "class_name", "pixels", "percentage"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for region_name, counts in [
            ("all_valid_pixels", prediction_stats["all_pixels"]),
            ("ood_pixels", prediction_stats["ood_pixels"]),
            ("in_distribution_pixels", prediction_stats["ind_pixels"]),
        ]:
            total = int(counts.sum())
            for row in class_count_rows(counts, total=total):
                writer.writerow({"region": region_name, **row})

    per_image_path = output_dir / "prediction_ood_summary_per_image.csv"
    with open(per_image_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "valid_pixels",
            "ood_pixels",
            "dominant_ood_predicted_class_id",
            "dominant_ood_predicted_class_name",
            "dominant_ood_predicted_percentage",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_stats["per_image_rows"])

    return aggregate_path, per_image_path


def print_prediction_stats(prediction_stats, top_k=10):
    """Stampa una sintesi compatta delle classi predette, soprattutto sui pixel OOD."""
    print("\nStatistiche predizioni semantic EoMT")
    print(f"Immagini processate: {prediction_stats['num_images']}")
    print(f"Immagini con pixel OOD: {prediction_stats['num_images_with_ood']}")

    for title, counts in [
        ("Classi predette su tutti i pixel validi", prediction_stats["all_pixels"]),
        ("Classi predette dentro le anomalie OOD", prediction_stats["ood_pixels"]),
    ]:
        total = int(counts.sum())
        print(f"\n{title}:")
        if total == 0:
            print("  nessun pixel disponibile")
            continue
        for row in class_count_rows(counts, total=total)[:top_k]:
            if row["pixels"] == 0:
                continue
            print(
                f"  {row['class_name']:<15} "
                f"{row['pixels']:>10d} px  {row['percentage']:>6.2f}%"
            )


def process_image(
    image_path,
    model,
    device,
    output_dir,
    metric_storage,
    prediction_stats,
    save_scores=True,
    compute_metrics=True,
):
    """Esegue inferenza, salva visualizzazioni e aggiorna statistiche."""
    image_tensor, pixel_logits = compute_pixel_logits(image_path, model, device)

    prediction = compute_semantic_prediction(pixel_logits)
    ood_gt = load_ood_ground_truth(image_path)
    score_maps = compute_anomaly_score_maps(pixel_logits)

    output_dir = Path(output_dir)
    image_stem = Path(image_path).stem

    prediction_path = output_dir / f"{image_stem}_prediction_vs_gt.png"
    plot_prediction_vs_gt(
        image_tensor=image_tensor,
        prediction=prediction,
        ood_gt=ood_gt,
        save_path=prediction_path,
    )

    score_path = None
    if save_scores:
        score_path = output_dir / f"{image_stem}_anomaly_scores.png"
        plot_anomaly_scores(
            image_tensor=image_tensor,
            score_maps=score_maps,
            save_path=score_path,
        )

    update_prediction_stats(image_path, prediction, ood_gt, prediction_stats)

    if compute_metrics:
        add_metrics_for_image(ood_gt, score_maps, metric_storage)

    del pixel_logits
    if device == "cuda":
        torch.cuda.empty_cache()

    return prediction_path, score_path


def collect_image_paths(input_pattern):
    """Accetta sia un path singolo sia un glob pattern."""
    expanded = os.path.expanduser(str(input_pattern))

    if os.path.isfile(expanded):
        return [expanded]

    return sorted(glob.glob(expanded))


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, help="Path singolo o glob delle immagini da processare.")
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/ml_anomaly_segmentation/visualizations",
        help="Cartella in cui salvare le visualizzazioni.",
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
        help="Se presente, salva solo prediction-vs-GT e non le mappe anomaly.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Se presente, non calcola/stampa AUPRC e FPR@TPR95.",
    )
    parser.add_argument(
        "--no-prediction-stats",
        action="store_true",
        help="Se presente, non stampa e non salva le statistiche sulle classi predette.",
    )
    parser.add_argument(
        "--prediction-stats-top-k",
        type=int,
        default=10,
        help="Numero di classi da stampare nelle statistiche delle predizioni.",
    )
    args = parser.parse_args()

    model, device = load_eomt_model(
        config_path=args.config_path,
        state_dict_path=args.state_dict_path,
        device_argument=args.device,
    )

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata con input: {args.input}")

    metric_storage = create_metric_storage()
    prediction_stats = create_prediction_stats()

    for image_path in image_paths:
        print(f"Processo: {image_path}")
        prediction_path, score_path = process_image(
            image_path=image_path,
            model=model,
            device=device,
            output_dir=args.output_dir,
            metric_storage=metric_storage,
            prediction_stats=prediction_stats,
            save_scores=not args.no_score_plots,
            compute_metrics=not args.no_metrics,
        )
        print(f"  Prediction vs GT salvata in: {prediction_path}")
        if score_path is not None:
            print(f"  Anomaly scores salvati in: {score_path}")

    if not args.no_prediction_stats:
        aggregate_csv, per_image_csv = save_prediction_stats_csv(prediction_stats, args.output_dir)
        print_prediction_stats(prediction_stats, top_k=args.prediction_stats_top_k)
        print(f"\nStatistiche predizioni salvate in: {aggregate_csv}")
        print(f"Sintesi OOD per immagine salvata in: {per_image_csv}")

    if not args.no_metrics:
        print_metric_results(metric_storage)


if __name__ == "__main__":
    main()
