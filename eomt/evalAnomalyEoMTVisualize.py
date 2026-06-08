"""
Visualizzazione di anomaly segmentation con EoMT.

Lo script fa tre cose:
1. esegue EoMT sulle immagini in input;
2. salva visualizzazioni di immagine, predizione semantica e ground truth OOD;
3. salva overlay immagine + anomaly heatmap.

Non calcola metriche globali, metriche per classe, CSV o curve diagnostiche.

Richiede un file functions.py che definisca:
- load_eomt
- eomt_to_pixel_logits
- load_ood_gt
- anomaly_scores
"""

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


def process_image(
    image_path,
    model,
    device,
    output_dir,
    save_overlay=True,
    overlay_score=DEFAULT_OVERLAY_SCORE,
):
    """Processa una immagine e salva output visuali."""
    image_tensor, pixel_logits = compute_pixel_logits(image_path, model, device)
    prediction = compute_semantic_prediction(pixel_logits)
    ood_gt = load_ood_ground_truth(image_path)
    score_maps = compute_anomaly_score_maps(pixel_logits)

    output_dir = Path(output_dir)
    image_stem = Path(image_path).stem

    prediction_path = output_dir / f"{image_stem}_prediction_vs_gt.pdf"
    plot_prediction_vs_gt(image_tensor, prediction, ood_gt, prediction_path)

    overlay_path = None
    if save_overlay:
        for score_name in SCORE_NAMES:
            overlay_path = (
                output_dir /
                f"{image_stem}_overlay_{score_name}.pdf"
            )

            plot_anomaly_overlay(
                image_tensor,
                score_maps[score_name],
                score_name,
                overlay_path,
            )

    del pixel_logits
    if device == "cuda":
        torch.cuda.empty_cache()

    return prediction_path, overlay_path


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
        help="Cartella in cui salvare visualizzazioni e overlay.",
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
        help="Score da usare per overlay.",
    )
    parser.add_argument(
        "--no-overlay-plots",
        action="store_true",
        help="Se presente, non salva gli overlay immagine + anomaly score.",
    )
    return parser


def main():
    """Esegue la pipeline di inferenza e visualizzazione."""
    parser = build_argument_parser()
    args = parser.parse_args()

    model, device = load_eomt_model(args.config_path, args.state_dict_path, args.device)

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata con input: {args.input}")

    for image_path in image_paths:
        print(f"Processo: {image_path}")
        prediction_path, overlay_path = process_image(
            image_path=image_path,
            model=model,
            device=device,
            output_dir=args.output_dir,
            save_overlay=not args.no_overlay_plots,
            overlay_score=args.overlay_score,
        )
        print(f"  Prediction vs GT salvata in: {prediction_path}")
        if overlay_path is not None:
            print(f"  Overlay salvato in: {overlay_path}")


if __name__ == "__main__":
    main()
