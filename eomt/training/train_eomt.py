## train_oe_eomt.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import numpy as np
import yaml
import warnings

from argparse import ArgumentParser

from functions import (
    load_eomt,
    freeze_model_except_final_parts,
    ood_hinge_loss,
    eomt_to_pixel_logits_train,
)

from datasets.cityscapes_semantic import CityscapesSemanticOE

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    lambda_oe=0.1,
    margin=0.1,
):
    model.train()

    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        images, targets = batch

        optimizer.zero_grad()

        batch_losses = []

        for image, target in zip(images, targets):
            image = image.to(device)

            if image.dtype != torch.uint8:
                image_input = (image * 255).to(torch.uint8)
            else:
                image_input = image

            logits = eomt_to_pixel_logits_train(
                image_input,
                device,
                model,
            )  # [19, H, W]

            ood_mask = target["ood_mask"].to(device)  # [H, W]

            loss_ood = ood_hinge_loss(
                logits=logits,
                ood_mask=ood_mask,
                margin=margin,
            )

            batch_losses.append(loss_ood)

        loss_ood_batch = torch.stack(batch_losses).mean()

        loss = lambda_oe * loss_ood_batch

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            print(
                f"batch {batch_idx:04d} | "
                f"loss={loss.item():.6f} | "
                f"loss_ood={loss_ood_batch.item():.6f}"
            )

    return epoch_loss / max(num_batches, 1)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--cityscapes-path",
        type=str,
        required=True,
        help="Path alla cartella Cityscapes",
    )

    parser.add_argument(
        "--coco-root",
        type=str,
        required=True,
        help="Path alla cartella COCO",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/eomt_cityscapes.bin",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="/content/drive/MyDrive/eomt_cityscapes_oe_finetuned.pth",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--p-ood", type=float, default=0.5)
    parser.add_argument("--lambda-oe", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.1)

    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Device:", device)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("Loading model...")
    model = load_eomt(
        device=device,
        config=config,
        state_dict_path=args.checkpoint,
    )

    model.to(device)

    print("Freezing model...")
    freeze_model_except_final_parts(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
            lambda_oe=args.lambda_oe,
            margin=args.margin,
        )

        print(f"Epoch {epoch + 1}/{args.epochs} | avg_loss={avg_loss:.6f}")

        torch.save(
            model.state_dict(),
            args.save_path,
        )

        print(f"Checkpoint saved to: {args.save_path}")

    print("Training completed.")


if __name__ == "__main__":
    main()
