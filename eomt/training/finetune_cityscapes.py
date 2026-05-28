import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import numpy as np
import yaml
import warnings

from argparse import ArgumentParser

from functions import *
from eomt.datasets.cityscapes_semantic import CityscapesSemanticOE
from eomt.training.mask_classification_loss import MaskClassificationLoss

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    alpha=5.0,
    file=None,
):
    model.train()

    epoch_loss = 0.0
    epoch_loss_eomt = 0.0
    epoch_loss_ood = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        images, targets = batch
        optimizer.zero_grad()
        images = images.to(device)

        if images.dtype != torch.uint8:
            images_input = (images * 255).to(torch.uint8)
        else:
            images_input = images

        targets_eomt = [
            {
                "masks": target["masks"].to(device).bool(),
                "labels": target["labels"].to(device).long(),
            }
            for target in targets
        ]

        ood_masks = torch.stack(
            [target["ood_mask"].to(device).bool() for target in targets],
            dim=0,
        )

        loss, loss_eomt, loss_ood, logits = eomt_forward_train_with_losses(
            model=model,
            imgs=images_input,
            targets_eomt=targets_eomt,
            ood_masks=ood_masks,
            device=device,
            alpha=alpha,
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_loss_eomt += loss_eomt.item()
        epoch_loss_ood += loss_ood.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            msg = (
                f"batch {batch_idx:04d} | "
                f"loss={loss.item():.6f} | "
                f"loss_eomt={loss_eomt.item():.6f} | "
                f"loss_ood={loss_ood.item():.6f}"
            )
            print(msg)

            if file is not None:
                file.write(msg + "\n")
                file.flush()

    return {
        "loss": epoch_loss / max(num_batches, 1),
        "loss_eomt": epoch_loss_eomt / max(num_batches, 1),
        "loss_ood": epoch_loss_ood / max(num_batches, 1),
    }

def main():
    parser = ArgumentParser()

    parser.add_argument("--cityscapes-path",
        type=str,
        required=True,
        help="/content/cityscapes/",
    )

    parser.add_argument(
        "--coco-root",
        type=str,
        required=True,
        help="/content/drive/MyDrive/cityscapes/coco",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="/content/drive/MyDrive/eomt_cityscapes_oe_finetuned.pth",
    ) # dove mettere i pesi aggiornati dopo finetuning

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    parser.add_argument("--p-ood", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=5.0)

    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    results_path = '/content/drive/MyDrive/results_finetune.txt'

    print("Scrivo risultati in:", results_path)

    file = open(results_path, 'w')
    file.flush()
    
    config_path = '../configs/dinov2/cityscapes/semantic/eomt_base_640.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    state_dict_path = '/content/drive/MyDrive/eomt_cityscapes.bin'
    
    warnings.filterwarnings("ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )
    
    # carica il modello
    model = load_eomt(device, config, state_dict_path)
    model.criterion.mask_coefficient = 5.0
    model.criterion.dice_coefficient = 5.0
    model.criterion.class_coefficient = 2.0
    model.criterion.eos_coef = 0.1

    model.to(device)

    print("Freezing model...")
    freeze_model_except_final_parts(model.network)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # l'optimizer prende i parametri solo non frizzati

    print("Preparing dataset...")

    data_module = CityscapesSemanticOE(
        path=args.cityscapes_path,
        coco_root=args.coco_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        p_ood=args.p_ood,
        img_size=(1024, 1024),
        check_empty_targets=True,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()

    print("Starting OE fine-tuning...")

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=args.alpha,
            file=file,
        )

        msg = (
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss['loss']:.6f} | "
            f"loss_seg={avg_loss['loss_eomt']:.6f} | "
            f"loss_ood={avg_loss['loss_ood']:.6f}"
        )

        print(msg)

        file.write(msg + "\n")
        file.flush()
        
        torch.save(
            model.state_dict(),
            args.save_path,
        )

        print(f"Checkpoint saved to: {args.save_path}")

    print("Training completed.")
    file.close()


if __name__ == "__main__":
    main()
