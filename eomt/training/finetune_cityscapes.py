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
    ignore_index=255,
):
    model.train()

    epoch_loss = 0.0
    epoch_loss_seg = 0.0
    epoch_loss_ood = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        images, targets = batch

        optimizer.zero_grad()

        # liste per le loss del batch
        batch_losses = []
        batch_seg_losses = []
        batch_ood_losses = []

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

            logits_b = logits.unsqueeze(0)  # [1, 19, H, W] per cross entropy

            # maschera semantica ID: [H, W], valori 0..18, ignore_index su pixel da ignorare
            sem_mask = target["mask"].to(device).long()  # oppure target["semantic_mask"]
            sem_mask_b = sem_mask.unsqueeze(0)           # [1, H, W]

            # maschera OoD: [H, W], bool
            ood_mask = target["ood_mask"].to(device).bool()

            # loss di segmentazione sui pixel ID
            loss_seg = F.cross_entropy(
                logits_b,
                sem_mask_b,
                ignore_index=ignore_index,
            )

            # loss OoD sui pixel outlier
            loss_ood = ood_hinge_loss(
                logits=logits,
                ood_mask=ood_mask,
                margin=margin,
            )

            loss = loss_seg + lambda_oe * loss_ood

            batch_losses.append(loss)
            batch_seg_losses.append(loss_seg)
            batch_ood_losses.append(loss_ood)

        loss_batch = torch.stack(batch_losses).mean()
        loss_seg_batch = torch.stack(batch_seg_losses).mean()
        loss_ood_batch = torch.stack(batch_ood_losses).mean()

        loss_batch.backward()
        optimizer.step()

        epoch_loss += loss_batch.item()
        epoch_loss_seg += loss_seg_batch.item()
        epoch_loss_ood += loss_ood_batch.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            print(
                f"batch {batch_idx:04d} | "
                f"loss={loss_batch.item():.6f} | "
                f"loss_seg={loss_seg_batch.item():.6f} | "
                f"loss_ood={loss_ood_batch.item():.6f}"
            )

    return {
        "loss": epoch_loss / max(num_batches, 1),
        "loss_seg": epoch_loss_seg / max(num_batches, 1),
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

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    config_path = 'eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    state_dict_path = '/content/drive/MyDrive/eomt_cityscapes.bin'
    
    warnings.filterwarnings("ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )
    
    # carica il modello
    model = load_eomt(device, config, state_dict_path)

    model.to(device)

    print("Freezing model...")
    freeze_model_except_final_parts(model)

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
            lambda_oe=args.lambda_oe,
            margin=args.margin,
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss['loss']:.6f} | "
            f"loss_seg={avg_loss['loss_seg']:.6f} | "
            f"loss_ood={avg_loss['loss_ood']:.6f}"
        )

        torch.save(
            model.state_dict(),
            args.save_path,
        )

        print(f"Checkpoint saved to: {args.save_path}")

    print("Training completed.")


if __name__ == "__main__":
    main()
