# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
    ):
        # inizializza la classe base nn.Module
        nn.Module.__init__(self)

        # numero di punti per calcolare la mask loss
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # coefficienti moltiplicativi (pesi) per bilanciare le diverse componenti della loss finale
        self.mask_coefficient = mask_coefficient # peso per la Sigmoid Cross Entropy Loss
        self.dice_coefficient = dice_coefficient # peso per la DICE Loss
        self.class_coefficient = class_coefficient # peso per la Cross Entropy Loss

        self.num_labels = num_labels # numero di classi reali del dataset
        self.eos_coef = no_object_coefficient # peso associato alla classe "nessun oggetto"

        # crea un vettore di pesi per le classi
        empty_weight = torch.ones(self.num_labels + 1)
        # assegna il coefficiente no_object
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # associa i pesi/predizioni della rete (queries) con gli oggetti reali presenti (ground truth)
        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        # estrae le maschere reali dai target per ogni immagine nel batch
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        # estrae le etichette di classe reali per ogni oggetto
        class_labels = [target["labels"].long() for target in targets]

        # esegue il matching biunivoco tra le query predette dalla rete e i target reali
        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        # calcola le loss relative alle maschere
        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        # calcola la loss di classificazione
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        return {**loss_masks, **loss_classes}

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        # conta il numero totale di maschere presenti nel batch corrente
        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        # se l'addestramento è distribuito su più GPU, sincronizza il numero di maschere
        if dist.is_available() and dist.is_initialized():
            # somma i tensori di tutte le GPU in modo che ognuna conosca il totale globale
            dist.all_reduce(num_masks_tensor)
            # recupera il numero totale di GPU attive nell'addestramento
            world_size = dist.get_world_size()
        else:
            world_size = 1

        # calcola la media delle maschere per GPU
        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        # normalizza ogni componente della loss delle maschere
        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None

        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            # controlla la tipologia di loss tramite la chiave e applica il rispettivo peso
            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss # Se è la prima loss della lista, la assegna direttamente
            else:
                # somma la loss corrente a quelle accumulate in precedenza
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  
