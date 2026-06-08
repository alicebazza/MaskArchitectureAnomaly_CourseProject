"""
Visualizzazione ed evaluation di anomaly segmentation con EoMT.

Versione diagnostica e commentata.

Lo script fa cinque cose:
1. esegue EoMT sulle immagini in input;
2. salva visualizzazioni di immagine, predizione semantica e ground truth OOD;
3. salva mappe di anomaly score e overlay immagine + anomaly heatmap;
4. calcola metriche OOD globali e per classe semantica predetta;
5. salva plot diagnostici globali: istogrammi ID/OOD, PR curve, ROC curve,
   ranking AUPRC per classe e rapporto OOD per classe.

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

try:
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
except ImportError:
    precision_recall_curve = None
    roc_curve = None
    auc = None

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
DEFAULT_OVERLAY_SCORE = "rba"

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
    """Sceglie il device richiesto oppure CUDA se disponibile."""
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
    """Carica una immagine RGB nel formato atteso da EoMT."""
    original_image = Image.open(image_path).convert("RGB")
    image_tensor = INPUT_TRANSFORM(original_image).float()
    image_tensor = (image_tensor * 255).to(torch.uint8)
    return image_tensor.to(device)


def compute_pixel_logits(image_path, model, device):
    """Esegue inferenza EoMT su una singola immagine."""
    image_tensor = load_image(image_path, device)
    with torch.no_grad():
        pixel_logits = eomt_to_pixel_logits(image_tensor, device, model)
    return image_tensor, pixel_logits


def compute_semantic_prediction(pixel_logits):
    """Converte i logits in una maschera semantica Cityscapes."""
    probabilities = F.softmax(pixel_logits.detach().cpu(), dim=0)
    return torch.argmax(probabilities, dim=0).numpy().astype(np.uint8)


def compute_anomaly_score_maps(pixel_logits):
    """Calcola le mappe MSP, MaxLogit, Entropy e RBA."""
    msp, maxlogit, entropy, rba = anomaly_scores(pixel_logits.detach().cpu(), use_rba=True)
    return {
        "msp": msp.detach().cpu().numpy(),
        "maxlogit": maxlogit.detach().cpu().numpy(),
        "entropy": entropy.detach().cpu().numpy(),
        "rba": rba.detach().cpu().numpy(),
    }


def load_ood_ground_truth(image_path):
    """Carica la ground truth OOD binaria associata alla immagine."""
    return load_ood_gt(image_path, size=IMAGE_SIZE)


def cityscapes_mapping():
    """Costruisce il dizionario classe Cityscapes -> colore RGB."""
    return {
        class_id: CITYSCAPES_PALETTE[class_id].astype(np.float32) / 255.0
        for class_id in range(len(CITYSCAPES_CLASSES))
    }


def apply_colormap(mask, mapping):
    """Trasforma una maschera intera in una immagine RGB."""
    colored = np.zeros((*mask.shape, 3), dtype=np.float32)
    for class_id in np.unique(mask):
        colored[mask == class_id] = mapping.get(int(class_id), [0.0, 0.0, 0.0])
    return colored


def tensor_to_numpy_image(image_tensor):
    """Converte un tensore immagine PyTorch in array NumPy visualizzabile."""
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    return np.clip(image_np, 0.0, 1.0)


def normalize_map(score_map):
    """Normalizza una mappa in [0, 1] solo per visualizzazione."""
    score_min = float(np.nanmin(score_map))
    score_max = float(np.nanmax(score_map))
    if score_max <= score_min:
        return np.zeros_like(score_map, dtype=np.float32)
    return ((score_map - score_min) / (score_max - score_min)).astype(np.float32)


def ensure_parent_dir(save_path):
    """Crea la cartella padre di un file di output."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def concatenate_metric_arrays(arrays):
    """Concatena liste di array 1D usate dagli accumulatori metrici."""
    if len(arrays) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate([np.asarray(x).reshape(-1) for x in arrays])


def plot_prediction_vs_gt(image_tensor, prediction, ood_gt, save_path):
    """Salva immagine, predizione semantica e ground truth OOD."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tensor_to_numpy_image(image_tensor))
    axes[0].set_title("Image")
    axes[1].imshow(apply_colormap(prediction, cityscapes_mapping()))
    axes[1].set_title("EoMT semantic prediction")
    axes[2].imshow(apply_colormap(ood_gt, OOD_PALETTE))
    axes[2].set_title("OOD ground truth")

    for ax in axes:
        ax.axis("off")

    ensure_parent_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_anomaly_scores(image_tensor, score_maps, save_path):
    """Salva immagine originale e mappe dei quattro anomaly score."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(tensor_to_numpy_image(image_tensor))
    axes[0].set_title("Image")
    axes[0].axis("off")

    for ax, score_name in zip(axes[1:], SCORE_NAMES):
        im = ax.imshow(score_maps[score_name])
        ax.set_title(score_name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ensure_parent_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_anomaly_overlay(image_tensor, score_map, score_name, save_path, alpha=0.45):
    """Salva un overlay immagine + heatmap per localizzare lo score."""
    image_np = tensor_to_numpy_image(image_tensor)
    normalized_score = normalize_map(score_map)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(image_np)
    im = ax.imshow(normalized_score, alpha=alpha)
    ax.set_title(f"Overlay anomaly score: {score_name}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ensure_parent_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def create_metric_storage():
    """Inizializza accumulatori globali e per classe predetta."""
    return {
        "global": {
            "ood_gts": [],
            "scores": {score_name: [] for score_name in SCORE_NAMES},
            "valid_pixels": 0,
            "ood_pixels": 0,
            "id_pixels": 0,
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
    """Aggiunge GT, score e conteggi pixel agli accumulatori."""
    valid_mask = ood_gt != IGNORE_INDEX
    if not np.any(valid_mask):
        return

    binary_gt = (ood_gt == 1).astype(np.uint8)
    valid_gt = binary_gt[valid_mask]

    metric_storage["global"]["valid_pixels"] += int(valid_mask.sum())
    metric_storage["global"]["ood_pixels"] += int(valid_gt.sum())
    metric_storage["global"]["id_pixels"] += int(valid_mask.sum() - valid_gt.sum())

    metric_storage["global"]["ood_gts"].append(valid_gt)
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

        class_storage["ood_gts"].append(class_gt)
        for score_name in SCORE_NAMES:
            class_storage["scores"][score_name].append(score_maps[score_name][class_mask])


def safe_eval_score(ood_gts, scores):
    """Calcola AUPRC e FPR@TPR95 gestendo casi non valutabili."""
    gt = concatenate_metric_arrays(ood_gts)
    if gt.size == 0 or np.unique(gt).size < 2:
        return None, None
    try:
        return eval_score(ood_gts, scores)
    except ValueError:
        return None, None


def compute_score_metrics(ood_gts, scores):
    """Restituisce metriche principali per una lista di GT e score."""
    auprc, fpr = safe_eval_score(ood_gts, scores)
    return {
        "auprc": auprc,
        "fpr_at_tpr95": fpr,
    }


def print_global_metrics(metric_storage):
    """Stampa metriche OOD globali per ogni anomaly score."""
    global_storage = metric_storage["global"]
    print("\nMetriche globali OOD")
    print(
        f"pixel validi: {global_storage['valid_pixels']} | "
        f"OOD: {global_storage['ood_pixels']} | ID: {global_storage['id_pixels']}"
    )

    for score_name in SCORE_NAMES:
        metrics = compute_score_metrics(
            global_storage["ood_gts"],
            global_storage["scores"][score_name],
        )
        if metrics["auprc"] is None:
            print(f"{score_name}: non calcolabile")
        else:
            print(f"AUPRC {score_name}: {metrics['auprc'] * 100.0:.4f}")
            print(f"FPR@TPR95 {score_name}: {metrics['fpr_at_tpr95'] * 100.0:.4f}")


def save_global_metrics_csv(metric_storage, output_dir):
    """Salva un CSV sintetico con metriche globali e conteggi pixel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ood_metrics_global.csv"
    global_storage = metric_storage["global"]

    fieldnames = [
        "score_name", "valid_pixels", "ood_pixels", "id_pixels", "ood_ratio",
        "auprc", "fpr_at_tpr95",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        valid_pixels = global_storage["valid_pixels"]
        ood_ratio = global_storage["ood_pixels"] / valid_pixels if valid_pixels > 0 else ""

        for score_name in SCORE_NAMES:
            metrics = compute_score_metrics(
                global_storage["ood_gts"],
                global_storage["scores"][score_name],
            )
            writer.writerow({
                "score_name": score_name,
                "valid_pixels": valid_pixels,
                "ood_pixels": global_storage["ood_pixels"],
                "id_pixels": global_storage["id_pixels"],
                "ood_ratio": ood_ratio,
                "auprc": "" if metrics["auprc"] is None else metrics["auprc"],
                "fpr_at_tpr95": "" if metrics["fpr_at_tpr95"] is None else metrics["fpr_at_tpr95"],
            })

    print("Salvo CSV globale in:", csv_path)
    return csv_path


def save_class_metrics_csv(metric_storage, output_dir):
    """Salva metriche per classe con conteggi e rapporto OOD."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ood_metrics_by_predicted_class.csv"

    fieldnames = [
        "class_id", "class_name", "valid_pixels", "ood_pixels", "id_pixels", "ood_ratio",
    ]
    for score_name in SCORE_NAMES:
        fieldnames.extend([f"{score_name}_auprc", f"{score_name}_fpr_at_tpr95"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for class_id, class_storage in metric_storage["by_predicted_class"].items():
            if class_storage["valid_pixels"] == 0:
                continue

            valid_pixels = class_storage["valid_pixels"]
            row = {
                "class_id": class_id,
                "class_name": CITYSCAPES_CLASSES[class_id],
                "valid_pixels": valid_pixels,
                "ood_pixels": class_storage["ood_pixels"],
                "id_pixels": class_storage["id_pixels"],
                "ood_ratio": class_storage["ood_pixels"] / valid_pixels if valid_pixels > 0 else "",
            }

            for score_name in SCORE_NAMES:
                metrics = compute_score_metrics(
                    class_storage["ood_gts"],
                    class_storage["scores"][score_name],
                )
                row[f"{score_name}_auprc"] = "" if metrics["auprc"] is None else metrics["auprc"]
                row[f"{score_name}_fpr_at_tpr95"] = (
                    "" if metrics["fpr_at_tpr95"] is None else metrics["fpr_at_tpr95"]
                )

            writer.writerow(row)

    print("Salvo CSV per classe in:", csv_path)
    return csv_path


def print_class_metrics(metric_storage):
    """Stampa una sintesi leggibile delle metriche per classe predetta."""
    print("\nMetriche OOD per classe semantica predetta")
    for class_id, class_storage in metric_storage["by_predicted_class"].items():
        if class_storage["valid_pixels"] == 0:
            continue

        print(f"\n{CITYSCAPES_CLASSES[class_id]}:")
        print(
            f"  pixel validi: {class_storage['valid_pixels']} | "
            f"OOD: {class_storage['ood_pixels']} | ID: {class_storage['id_pixels']} | "
            f"OOD ratio: {class_storage['ood_pixels'] / class_storage['valid_pixels']:.6f}"
        )

        for score_name in SCORE_NAMES:
            metrics = compute_score_metrics(
                class_storage["ood_gts"],
                class_storage["scores"][score_name],
            )
            if metrics["auprc"] is not None:
                print(
                    f"  {score_name:<8} "
                    f"AUPRC={metrics['auprc'] * 100.0:7.4f}  "
                    f"FPR@TPR95={metrics['fpr_at_tpr95'] * 100.0:7.4f}"
                )
            else:
                print(f"  {score_name:<8} non calcolabile")


def plot_score_histograms(metric_storage, output_dir, bins=100):
    """Salva istogrammi degli score separando pixel ID e OOD."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = concatenate_metric_arrays(metric_storage["global"]["ood_gts"]).astype(bool)
    if gt.size == 0 or np.unique(gt).size < 2:
        print("Istogrammi non calcolabili: servono pixel ID e OOD.")
        return None

    save_path = output_dir / "score_histograms_id_vs_ood.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.reshape(-1)

    for ax, score_name in zip(axes, SCORE_NAMES):
        scores = concatenate_metric_arrays(metric_storage["global"]["scores"][score_name])
        ax.hist(scores[~gt], bins=bins, density=True, alpha=0.5, label="ID")
        ax.hist(scores[gt], bins=bins, density=True, alpha=0.5, label="OOD")
        ax.set_title(score_name)
        ax.set_xlabel("anomaly score")
        ax.set_ylabel("density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Salvo istogrammi ID/OOD in:", save_path)
    return save_path


def plot_pr_curves(metric_storage, output_dir):
    """Salva le Precision-Recall curve globali per tutti gli score."""
    if precision_recall_curve is None or auc is None:
        print("PR curve non salvata: scikit-learn non disponibile.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = concatenate_metric_arrays(metric_storage["global"]["ood_gts"])
    if gt.size == 0 or np.unique(gt).size < 2:
        print("PR curve non calcolabile: servono pixel ID e OOD.")
        return None

    save_path = output_dir / "pr_curves_global.png"
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for score_name in SCORE_NAMES:
        scores = concatenate_metric_arrays(metric_storage["global"]["scores"][score_name])
        precision, recall, _ = precision_recall_curve(gt, scores)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{score_name} AUPRC={pr_auc:.4f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Global Precision-Recall curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Salvo PR curve in:", save_path)
    return save_path


def plot_roc_curves(metric_storage, output_dir):
    """Salva le ROC curve globali per tutti gli score."""
    if roc_curve is None or auc is None:
        print("ROC curve non salvata: scikit-learn non disponibile.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = concatenate_metric_arrays(metric_storage["global"]["ood_gts"])
    if gt.size == 0 or np.unique(gt).size < 2:
        print("ROC curve non calcolabile: servono pixel ID e OOD.")
        return None

    save_path = output_dir / "roc_curves_global.png"
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for score_name in SCORE_NAMES:
        scores = concatenate_metric_arrays(metric_storage["global"]["scores"][score_name])
        fpr, tpr, _ = roc_curve(gt, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{score_name} AUROC={roc_auc:.4f}")

    ax.plot([0, 1], [0, 1], linestyle="--", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Global ROC curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Salvo ROC curve in:", save_path)
    return save_path


def collect_class_values(metric_storage, score_name):
    """Raccoglie AUPRC e OOD ratio per le classi valutabili."""
    class_names = []
    auprcs = []
    ood_ratios = []
    valid_pixels = []

    for class_id, class_storage in metric_storage["by_predicted_class"].items():
        if class_storage["valid_pixels"] == 0:
            continue
        metrics = compute_score_metrics(
            class_storage["ood_gts"],
            class_storage["scores"][score_name],
        )
        class_names.append(CITYSCAPES_CLASSES[class_id])
        auprcs.append(np.nan if metrics["auprc"] is None else metrics["auprc"])
        ood_ratios.append(class_storage["ood_pixels"] / class_storage["valid_pixels"])
        valid_pixels.append(class_storage["valid_pixels"])

    return class_names, np.array(auprcs), np.array(ood_ratios), np.array(valid_pixels)


def plot_class_auprc(metric_storage, output_dir, score_name=DEFAULT_OVERLAY_SCORE):
    """Salva un ranking AUPRC per classe per uno score scelto."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names, auprcs, _, _ = collect_class_values(metric_storage, score_name)
    if len(class_names) == 0 or np.all(np.isnan(auprcs)):
        print("Ranking AUPRC per classe non calcolabile.")
        return None

    valid = ~np.isnan(auprcs)
    class_names = np.array(class_names)[valid]
    auprcs = auprcs[valid]
    order = np.argsort(auprcs)

    save_path = output_dir / f"class_auprc_ranking_{score_name}.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, max(5, 0.35 * len(class_names))))
    ax.barh(class_names[order], auprcs[order])
    ax.set_xlabel("AUPRC")
    ax.set_title(f"AUPRC per predicted class ({score_name})")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Salvo ranking AUPRC per classe in:", save_path)
    return save_path


def plot_class_ood_ratio(metric_storage, output_dir):
    """Salva il rapporto OOD/validi per ciascuna classe predetta."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names, _, ood_ratios, valid_pixels = collect_class_values(metric_storage, DEFAULT_OVERLAY_SCORE)
    if len(class_names) == 0:
        print("OOD ratio per classe non calcolabile.")
        return None

    class_names = np.array(class_names)
    order = np.argsort(ood_ratios)

    save_path = output_dir / "class_ood_ratio.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, max(5, 0.35 * len(class_names))))
    ax.barh(class_names[order], ood_ratios[order])
    ax.set_xlabel("OOD pixels / valid pixels")
    ax.set_title("OOD ratio per predicted class")
    ax.grid(True, axis="x", alpha=0.3)

    for y, idx in enumerate(order):
        ax.text(ood_ratios[idx], y, f"  n={valid_pixels[idx]}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Salvo OOD ratio per classe in:", save_path)
    return save_path


def save_diagnostic_plots(metric_storage, output_dir, class_score=DEFAULT_OVERLAY_SCORE):
    """Salva tutti i plot diagnostici globali e per classe."""
    paths = []
    for path in [
        plot_score_histograms(metric_storage, output_dir),
        plot_pr_curves(metric_storage, output_dir),
        plot_roc_curves(metric_storage, output_dir),
        plot_class_auprc(metric_storage, output_dir, score_name=class_score),
        plot_class_ood_ratio(metric_storage, output_dir),
    ]:
        if path is not None:
            paths.append(path)
    return paths


def process_image(
    image_path,
    model,
    device,
    output_dir,
    metric_storage,
    save_scores=True,
    save_overlay=True,
    overlay_score=DEFAULT_OVERLAY_SCORE,
):
    """Processa una immagine e aggiorna output visuali e metriche."""
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

    overlay_path = None
    if save_overlay:
        if overlay_score not in SCORE_NAMES:
            raise ValueError(f"overlay_score deve essere in {SCORE_NAMES}, ricevuto: {overlay_score}")
        overlay_path = output_dir / f"{image_stem}_overlay_{overlay_score}.png"
        plot_anomaly_overlay(image_tensor, score_maps[overlay_score], overlay_score, overlay_path)

    add_metrics_for_image(ood_gt, prediction, score_maps, metric_storage)

    del pixel_logits
    if device == "cuda":
        torch.cuda.empty_cache()

    return prediction_path, score_path, overlay_path


def collect_image_paths(input_pattern):
    """Restituisce una lista ordinata di immagini da processare."""
    expanded = os.path.expanduser(str(input_pattern))
    if os.path.isfile(expanded):
        return [expanded]
    return sorted(glob.glob(expanded))


def build_argument_parser():
    """Crea il parser degli argomenti da linea di comando."""
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, help="Path singolo o glob delle immagini da processare.")
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/ml_anomaly_segmentation/visualizations",
        help="Cartella in cui salvare visualizzazioni, CSV e plot diagnostici.",
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
        "--overlay-score",
        choices=SCORE_NAMES,
        default=DEFAULT_OVERLAY_SCORE,
        help="Score da usare per overlay e ranking per classe.",
    )
    parser.add_argument(
        "--no-score-plots",
        action="store_true",
        help="Se presente, non salva le quattro mappe anomaly per immagine.",
    )
    parser.add_argument(
        "--no-overlay-plots",
        action="store_true",
        help="Se presente, non salva gli overlay immagine + anomaly score.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Se presente, non calcola metriche globali e per classe.",
    )
    parser.add_argument(
        "--no-diagnostic-plots",
        action="store_true",
        help="Se presente, non salva istogrammi, PR/ROC curve e plot per classe.",
    )
    return parser


def main():
    """Esegue l'intera pipeline di inferenza, visualizzazione e valutazione."""
    parser = build_argument_parser()
    args = parser.parse_args()

    model, device = load_eomt_model(args.config_path, args.state_dict_path, args.device)

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata con input: {args.input}")

    metric_storage = create_metric_storage()

    for image_path in image_paths:
        print(f"Processo: {image_path}")
        prediction_path, score_path, overlay_path = process_image(
            image_path=image_path,
            model=model,
            device=device,
            output_dir=args.output_dir,
            metric_storage=metric_storage,
            save_scores=not args.no_score_plots,
            save_overlay=not args.no_overlay_plots,
            overlay_score=args.overlay_score,
        )
        print(f"  Prediction vs GT salvata in: {prediction_path}")
        if score_path is not None:
            print(f"  Anomaly scores salvati in: {score_path}")
        if overlay_path is not None:
            print(f"  Overlay salvato in: {overlay_path}")

    if not args.no_metrics:
        print_global_metrics(metric_storage)
        print_class_metrics(metric_storage)
        global_csv_path = save_global_metrics_csv(metric_storage, args.output_dir)
        class_csv_path = save_class_metrics_csv(metric_storage, args.output_dir)
        print(f"\nMetriche globali salvate in: {global_csv_path}")
        print(f"Metriche per classe salvate in: {class_csv_path}")

        if not args.no_diagnostic_plots:
            diagnostic_paths = save_diagnostic_plots(
                metric_storage,
                args.output_dir,
                class_score=args.overlay_score,
            )
            print("\nPlot diagnostici salvati:")
            for path in diagnostic_paths:
                print(f"  {path}")


if __name__ == "__main__":
    main()
