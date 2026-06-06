# ---------------------------------------------------------------
# OE extension for semantic mask classification.
# Compatible with the LightningCLI training flow in main.py.
# ---------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities import rank_zero_info

from training.mask_classification_semantic import MaskClassificationSemantic


class MaskClassificationSemanticOE(MaskClassificationSemantic):

    """
    Estensione di MaskClassificationSemantic che introduce
    Outlier Exposure (OE) per la segmentazione semantica.

    La classe aggiunge una loss OoD basata su RBA, combinata con la loss standard di segmentazione.
    È inoltre possibile congelare tutti i parametri del modello
    eccetto gli head di classificazione e segmentazione.

    Parametri aggiuntivi:
        - lambda_rba : peso della loss OoD nella loss totale
        - rba_alpha : margine della hinge loss OoD
        - rba_reduction: modalità di aggregazione della loss OoD ("mean" o "sum")
        - freeze_heads_only: se True, rende trainabili solo gli head del modello.
    """

    def __init__(
        self,
        *args,
        lambda_rba: float = 0.1,
        rba_alpha: float = 5.0,
        rba_reduction: str = "mean",
        freeze_heads_only: bool = False,
        **kwargs,
    ) -> None:
        # richiama il costruttore della classe base
        super().__init__(*args, **kwargs)

        # inizializza gli iperparametri specifici per il meccanismo RbA e Outlier Exposure
        self.lambda_rba = lambda_rba
        self.rba_alpha = rba_alpha
        self.rba_reduction = rba_reduction
        self.freeze_heads_only = freeze_heads_only

        # salva automaticamente gli iperparametri nel modulo di PyTorch Lightning
        self.save_hyperparameters(
            {
                "lambda_rba": lambda_rba,
                "rba_alpha": rba_alpha,
                "rba_reduction": rba_reduction,
                "freeze_heads_only": freeze_heads_only,
            }
        )

        # se richiesto, congela il resto della rete lasciando attivi solo i blocchi finali
        if freeze_heads_only:
            self.freeze_all_but_heads()
            self.print_trainable_parameters()


    @staticmethod
    def ood_hinge_loss(
        logits: torch.Tensor,
        ood_mask: torch.Tensor,
        alpha: float,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Cosa fa:
            Calcola una hinge loss quadratica sui pixel OoD.

        Input:
            - logits: tensore [B, 19, H, W]
            - ood_mask: tensore [B, H, W] con True/1 sui pixel OoD.
            - alpha: margine della hinge loss.
            - reduction:
              modalità di aggregazione della loss:
                  * "sum"  -> somma la loss di tutti i pixel OoD
                  * "mean" -> media la loss sui pixel OoD

        Output:
            -  loss OoD.
        """

        # trasferisce la maschera delle anomalie sulla stessa GPU/CPU dei logit e la forza a booleana
        ood_mask = ood_mask.to(logits.device).bool()

        # se nessuna immagine nel batch corrente contiene anomalie incollata da COCO, restituisce 0
        if not ood_mask.any():
            return logits.new_zeros(())

        score = torch.tanh(logits)
        rba = -score.sum(dim=1)

        # calcola la Hinge Loss quadratica: penalizza i pixel dove la confidenza di RbA è inferiore al margine alpha
        loss_map = F.relu(alpha - rba).pow(2)

        # estrae solo i valori di loss corrispondenti ai pixel contrassegnati come anomalie
        selected = loss_map[ood_mask]

        # riduce l'array di loss dei pixel in un unico scalare basandosi sulla strategia selezionata
        if reduction == "sum":
            return selected.sum()
        if reduction == "mean":
            return selected.mean()

        raise ValueError(f"Riduzione non supportata: {reduction}")

    def freeze_all_but_heads(self) -> None:
        """
        Congela tutti i parametri del modello e rende trainabili solo
        gli head di classificazione e segmentazione.

        Input: Nessuno.
        Output: Nessuno.

        """

        # disattiva il calcolo del gradiente
        for param in self.network.parameters():
            param.requires_grad = False

        for module_name in ("class_head", "mask_head"):

            module = getattr(self.network, module_name)

            # riattiva il calcolo dei gradienti esclusivamente per i pesi di questi due head finali
            for param in module.parameters():
                param.requires_grad = True

        trainable_params = [
            name
            for name, param in self.network.named_parameters()
            if param.requires_grad
        ]

        if not trainable_params:
            raise RuntimeError("Nessun parametro trainabile trovato dopo il freezing.")

    def print_trainable_parameters(self) -> None:
        """
        Stampa informazioni sui parametri trainabili del modello.

        Input: Nessuno.
        Output: Nessuno.

        """

        # calcola il volume totale di tutti i parametri presenti nell'architettura
        total_params = sum(p.numel() for p in self.network.parameters())
        # isola le componenti che hanno il gradiente attivo
        trainable_params = [
            (name, p) for name, p in self.network.named_parameters() if p.requires_grad
        ]
        # conta quanti sono i singoli pesi effettivamente aggiornabili
        trainable_count = sum(p.numel() for _, p in trainable_params)
        percentage = 100.0 * trainable_count / max(total_params, 1)

        rank_zero_info(
            f"Trainable parameters: {trainable_count:,}/{total_params:,} "
            f"({percentage:.2f}%)"
        )
        for name, _ in trainable_params:
            rank_zero_info(f"  - {name}")

    def _targets_without_oe_fields(
        self,
        targets: list[dict]
    ) -> list[dict]:
        """
        Cosa fa:
            Prepara i target per la loss di segmentazione standard,
            rimuovendo i campi aggiuntivi usati per l'Outlier Exposure.

        Input:
            - targets: lista di dizionari, uno per immagine del batch.

        Output:
            - Lista di dizionari contenenti esclusivamente:
                  * "masks" (convertite in bool)
                  * "labels" (convertite in long)
              pronti per essere utilizzati da MaskClassificationLoss.
        """
        cleaned = []

        for target in targets:
            cleaned.append(
                {
                    "masks": target["masks"].to(self.device).bool(),
                    "labels": target["labels"].to(self.device).long(),
                }
            )
        return cleaned

    def _batch_ood_masks(
        self,
        targets: list[dict],
        size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Costruisce un batch di maschere OOD (Out-Of-Distribution).

        Input:
            - targets: lista di dizionari, uno per elemento del batch
            - size: dimensione attesa della maschera

        Output:
            Tensore booleano di forma (B, H, W).
                Ogni elemento contiene la maschera OOD del corrispondente
                campione del batch.
        """

        masks = []

        for target in targets:
            mask = target.get("ood_mask")

            # se l'immagine corrente fa parte del campione pulito (senza Cut-Paste), crea una maschera di zeri
            if mask is None:
                mask = torch.zeros(
                    size,
                    device=self.device,
                    dtype=torch.bool,
                )
            else:
                mask = mask.to(self.device)

                if mask.ndim > 2:
                    mask = mask.squeeze()

                # se la maschera dell'anomalia ha una risoluzione diversa dall'immagine la corregge
                if tuple(mask.shape[-2:]) != tuple(size):
                    mask = F.interpolate(
                        mask[None, None].float(),
                        size=size,
                        mode="nearest",
                    )[0, 0]

                mask = mask.bool()
            masks.append(mask)

        # unisce tutte le singole maschere del batch: risultato [Batch, H, W]
        return torch.stack(masks, dim=0)


    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        # isola i target standard
        targets_eomt = self._targets_without_oe_fields(targets)
        ood_masks = self._batch_ood_masks(targets, size=imgs.shape[-2:])

        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        # cicla attraverso i layer intermedi dell'architettura
        for i, (mask_logits, class_logits) in enumerate(
            zip(mask_logits_per_block, class_logits_per_block)
        ):
            # calcola la loss di segmentazione
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets_eomt,
            )

            block_postfix = self.block_postfix(i)
            losses_all_blocks |= {
                f"{key}{block_postfix}": value for key, value in losses.items()
            }

        loss_eomt = self.criterion.loss_total(losses_all_blocks, self.log)

        # isola l'output geometrico generato dall'ultimo blocco della rete
        final_mask_logits = mask_logits_per_block[-1]

        if final_mask_logits.shape[-2:] != imgs.shape[-2:]:
            final_mask_logits = F.interpolate(
                final_mask_logits,
                size=imgs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # trasforma la rappresentazione interna basata su Query in una mappa logit classica pixel-per-pixel
        per_pixel_logits = self.to_per_pixel_logits_semantic(
            final_mask_logits,
            class_logits_per_block[-1],
        )

        # calcola la Hinge Loss basata sul principio matematico di RbA applicata sui soli pixel delle anomalie COCO
        loss_ood = self.ood_hinge_loss(
            per_pixel_logits,
            ood_masks,
            alpha=self.rba_alpha,
            reduction=self.rba_reduction,
        )

        loss_total = loss_eomt + self.lambda_rba * loss_ood

        self.log("train/loss_eomt", loss_eomt, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_ood", loss_ood, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_total", loss_total, on_step=True, on_epoch=True, prog_bar=True)

        self.log(
            "losses_epoch/train_loss_total_oe",
            loss_total,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "losses_epoch/train_rba_loss",
            loss_ood,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "losses_epoch/train_loss_without_rba",
            loss_eomt,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss_total
