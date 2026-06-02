# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------


import jsonargparse._typehints as _t
import os
from types import MethodType
from pathlib import Path
from gitignore_parser import parse_gitignore
import logging
import torch
import warnings
from lightning.pytorch import cli
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

# Suppress PyTorch FX warnings for DINOv3 models
import os
os.environ["TORCH_LOGS"] = "-dynamo"


def _default_run_root() -> Path:
    """
    Usa Drive su Colab se disponibile, per salvare log e checkpoint.
    """

    colab_drive = Path("/content/drive/MyDrive")
    if colab_drive.exists():
        return colab_drive / "eomt_runs"
    return Path.cwd() / "eomt_runs"


def _find_model_checkpoint_callback(trainer) -> ModelCheckpoint | None:
    """
    Cerca il callback ModelCheckpoint tra quelli registrati dal trainer.
    """

    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    return None


_orig_single = _t.raise_unexpected_value


def _raise_single(*args, exception=None, **kwargs):
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)


_orig_union = _t.raise_union_unexpected_value


def _raise_union(subtypes, val, vals):
    for e in reversed(vals):
        if isinstance(e, Exception):
            raise e
    return _orig_union(subtypes, val, vals)


_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union

# ridefinisce quando Lightning deve fare validazione
def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # added below to check val based on global steps instead of batches in case of iteration based val check and gradient accumulation
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
        )
        warnings.filterwarnings(
            "ignore", message=r"^Grad strides do not match bucket view strides.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*functools.partial will be a method descriptor in future Python versions*",
        )

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile_disabled", action="store_true")
        parser.add_argument(
            "--resume_disabled",
            action="store_true",
            help="Disabilita il resume automatico dall'ultimo checkpoint trovato.",
        )

        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )

        parser.link_arguments(
            "model.init_args.ckpt_path",
            "model.init_args.network.init_args.encoder.init_args.ckpt_path",
        )

    def fit(self, model, **kwargs):
        if hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        checkpoint_callback = _find_model_checkpoint_callback(self.trainer)
        if checkpoint_callback is not None:
            run_name = getattr(self.trainer.logger, "name", "default_run")
            checkpoint_dir = _default_run_root() / "checkpoints" / run_name
            checkpoint_callback.dirpath = str(checkpoint_dir)
            checkpoint_callback.FILE_EXTENSION = ".ckpt"

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        # Se esiste un last.ckpt nella cartella del checkpoint, ripartiamo da lì.
        if not self.config[self.config["subcommand"]]["resume_disabled"]:
            checkpoint_callback = _find_model_checkpoint_callback(self.trainer)
            if checkpoint_callback is not None and checkpoint_callback.dirpath is not None:
                last_checkpoint = Path(checkpoint_callback.dirpath) / "last.ckpt"
                if last_checkpoint.exists() and kwargs.get("ckpt_path") is None:
                    logging.info(f"Resuming from checkpoint: {last_checkpoint}")
                    kwargs["ckpt_path"] = str(last_checkpoint)

        self.trainer.fit(model, **kwargs)


def cli_main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "default_root_dir": str(_default_run_root()),
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    dirpath=str(_default_run_root() / "checkpoints" / "pending_run"),
                    filename="epoch{epoch:03d}-step{step:06d}",
                    save_last=True,
                    save_top_k=-1,
                    every_n_epochs=1,
                ),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    cli_main()
