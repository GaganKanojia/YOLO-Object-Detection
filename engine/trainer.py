import os
import math
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from copy import deepcopy
from tqdm import tqdm

from models.yolov11 import load_model
from data.loaders import build_dataloader
from data.dataset import build_dataset, parse_yolo_yaml
from loss.loss import DetectionLoss
from engine.validator import Validator
from utils.general import (
    ModelEMA, build_optimizer, linear_lr, one_cycle,
    increment_path, setup_logging
)

logger = setup_logging("trainer")


class EarlyStopping:
    def __init__(self, patience=100):
        self.patience = patience
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.best_epoch = epoch
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience
        if stop:
            logger.info(f"Early stopping at epoch {epoch}. Best epoch was {self.best_epoch}.")
        return stop


class Trainer:
    def __init__(self, cfg_model: str, cfg_train: str, data_cfg: dict, device="cuda"):
        with open(cfg_train) as f:
            self.args = yaml.safe_load(f)
        self.args.update(data_cfg)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.args.get("save_dir"):
            self.save_dir = increment_path(Path(self.args["save_dir"]), mkdir=True)
        else:
            self.save_dir = increment_path(
                Path(self.args.get("project", "runs/train")) / self.args.get("name", "exp"),
                mkdir=True
            )
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.cfg_model = cfg_model
        self.best_fitness = 0.0
        self.start_epoch = 0
        self.resume = None  # path to a checkpoint to resume from

    def _load_resume(self, model, ema, optimizer, scheduler, scaler, epochs):
        """Restore full training state from a checkpoint saved by this trainer."""
        ckpt = torch.load(self.resume, map_location=self.device, weights_only=False)

        if ckpt.get("model") is not None:
            model.load_state_dict(ckpt["model"], strict=False)
        if ckpt.get("ema") is not None:
            ema.ema.load_state_dict(ckpt["ema"], strict=False)
        ema.updates = ckpt.get("updates", ema.updates)
        if ckpt.get("optimizer") is not None:
            # Restores per-group LR/momentum (incl. the post-step LR for the resumed epoch).
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])  # no-op if AMP is disabled

        self.best_fitness = ckpt.get("best_fitness", 0.0)
        self.start_epoch = ckpt.get("epoch", -1) + 1
        # The saved optimizer already holds lf(start_epoch); align the scheduler so the
        # end-of-epoch step advances to lf(start_epoch + 1) rather than resetting to lf(1).
        scheduler.last_epoch = self.start_epoch
        logger.info(
            f"Resumed from {self.resume} → epoch {self.start_epoch + 1}/{epochs}, "
            f"best_fitness={self.best_fitness:.4f}, ema.updates={ema.updates}"
        )

    def train(self):
        args = self.args
        epochs = args["epochs"]
        batch_size = args["batch"]
        nbs = args.get("nbs", 64)
        imgsz = args["imgsz"]
        close_mosaic = args.get("close_mosaic", 10)

        # In YOLO format, nc is defined by data.yaml — derive it before building
        # the model so the detection head gets the right number of classes.
        if args.get("nc") is None and ("train_yaml" in args or "val_yaml" in args):
            yaml_file = args.get("train_yaml") or args.get("val_yaml")
            args["nc"] = parse_yolo_yaml(yaml_file)["nc"]
            logger.info(f"Inferred nc={args['nc']} from {yaml_file}")

        # Model
        model = load_model(self.cfg_model, nc=args.get("nc"), device=str(self.device))
        model.to(self.device).train()

        # Pretrained fine-tuning weights. Skipped when resuming — resume restores the
        # full training state (weights+optimizer+EMA) and must not be overwritten here.
        if args.get("pretrained_weights") and not self.resume:
            from utils.transfer import load_pretrained_weights, inspect_checkpoint
            pw = args["pretrained_weights"]
            pinfo = inspect_checkpoint(pw)
            logger.info(
                f"Pretrained: nc={pinfo['nc_pretrained']}, epoch={pinfo['epoch']}, "
                f"fitness={pinfo['best_fitness']:.4f}, params={pinfo['param_count']:.2f}M"
            )
            res = load_pretrained_weights(
                model=model, weights_path=pw, nc_new=args.get("nc"),
                strategy=args.get("strategy", "partial"),
                nc_subset_map=args.get("nc_subset_map"), verbose=True,
            )
            logger.info(
                f"Loaded {len(res['loaded_keys'])} keys, "
                f"reinitialized {len(res['reinitialized'])}, skipped {len(res['skipped_keys'])}"
            )

        # Optimizer
        accumulate = max(round(nbs / batch_size), 1)
        wd = args["weight_decay"] * batch_size * accumulate / nbs
        opt_name = "AdamW" if args["optimizer"] == "auto" else args["optimizer"]
        use_diff_lr = args.get("use_differential_lr", False)
        if use_diff_lr:
            from utils.transfer import build_param_groups, build_optimizer_from_groups
            param_groups = build_param_groups(
                model=model, lr_backbone=args["lr_backbone"], lr_neck=args["lr_neck"],
                lr_head_box=args["lr_head_box"], lr_head_cls_ft=args["lr_head_cls_ft"],
                lr_head_cls_new=args["lr_head_cls_new"], weight_decay=wd,
            )
            optimizer = build_optimizer_from_groups(param_groups, name=opt_name,
                                                    momentum=args["momentum"])
            logger.info(f"Differential LR: {len(optimizer.param_groups)} param groups")
        else:
            optimizer = build_optimizer(model, name=args["optimizer"], lr=args["lr0"],
                                        momentum=args["momentum"], decay=wd)

        # LR scheduler
        if args.get("cos_lr", False):
            lf = one_cycle(1, args["lrf"], epochs)
        else:
            lf = linear_lr(args["lrf"], epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # AMP
        scaler = GradScaler(enabled=args.get("amp", True) and self.device.type == "cuda")

        # EMA
        ema = ModelEMA(model)

        # Resume: restore weights, optimizer, EMA, scaler, and training position.
        # Must run after model/optimizer/scheduler/scaler/ema exist so their state
        # can be loaded in place.
        if self.resume:
            self._load_resume(model, ema, optimizer, scheduler, scaler, epochs)

        # Loss
        criterion = DetectionLoss(
            model,
            box=args.get("box", 7.5),
            cls=args.get("cls", 0.5),
            dfl=args.get("dfl", 1.5),
        )

        # Early stopping
        stopper = EarlyStopping(patience=args.get("patience", 100))

        # Paths for checkpoints (Ultralytics layout: runs/<save_dir>/weights/{last,best}.pt)
        last = self.weights_dir / "last.pt"
        best = self.weights_dir / "best.pt"
        results_csv = self.save_dir / "results.csv"
        with open(results_csv, "w") as f:
            f.write("epoch,box_loss,cls_loss,dfl_loss,total_loss,precision,recall,mAP50,mAP50_95\n")

        logger.info(f"Training for {epochs} epochs → {self.save_dir}")

        nb = None  # will be set once loader created
        last_opt_step = -1

        for epoch in range(self.start_epoch, epochs):
            is_final = epoch + 1 == epochs
            disable_mosaic = epoch >= (epochs - close_mosaic)

            # Backbone freeze/unfreeze schedule (fine-tuning). Freeze for the first
            # `freeze_backbone_epochs`, then unfreeze and (for the differential-LR
            # path) rebuild the optimizer so the now-trainable backbone params join
            # with fresh state, rebinding the scheduler to the new optimizer.
            if args.get("freeze_backbone"):
                from utils.transfer import freeze_backbone, rebuild_optimizer_after_unfreeze
                fe = args.get("freeze_backbone_epochs", 5)
                if epoch == self.start_epoch and epoch < fe:
                    frozen = freeze_backbone(model, freeze=True)
                    logger.info(f"Backbone frozen ({len(frozen)} params) until epoch {fe}")
                elif epoch == fe:
                    unfrozen = freeze_backbone(model, freeze=False)
                    logger.info(f"Backbone unfrozen at epoch {epoch} ({len(unfrozen)} params)")
                    if use_diff_lr:
                        optimizer = rebuild_optimizer_after_unfreeze(
                            model, {**args, "weight_decay": wd, "optimizer": opt_name}, optimizer
                        )
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
                        scheduler.last_epoch = epoch
                        for pg in optimizer.param_groups:
                            pg["lr"] = pg["initial_lr"] * lf(epoch)

            # Rebuild dataset+loader each epoch (to toggle the mosaic schedule).
            # build_dataset picks YOLODataset (COCO) or YOLOTxtDataset (YOLO) based
            # on which config keys are present. Mosaic-family augs are disabled for
            # the final `close_mosaic` epochs by overriding their probabilities.
            epoch_args = {
                **args,
                "mosaic": 0.0 if disable_mosaic else args.get("mosaic", 1.0),
                "mixup": 0.0 if disable_mosaic else args.get("mixup", 0.0),
                "cutmix": 0.0 if disable_mosaic else args.get("cutmix", 0.0),
                "copy_paste": 0.0 if disable_mosaic else args.get("copy_paste", 0.0),
                "epoch": epoch,
            }
            train_dataset = build_dataset(epoch_args, split="train", augment=True)
            train_loader, _ = build_dataloader(
                train_dataset,
                batch_size=batch_size,
                workers=args.get("workers", 8),
                augment=True,
                shuffle=True,
            )

            if nb is None:
                nb = len(train_loader)
                # Respect warmup_epochs literally — no 100-iter floor.
                # The old `max(..., 100)` floor pathologically dominated short
                # smoke runs (e.g. batch=32, nb=4 → 25 epochs of forced warmup).
                nw = max(round(args.get("warmup_epochs", 3.0) * nb), 1)

            model.train()
            optimizer.zero_grad()
            pbar = tqdm(enumerate(train_loader), total=nb, desc=f"Epoch {epoch+1}/{epochs}")

            mloss = torch.zeros(3, device=self.device)
            for i, batch in pbar:
                ni = i + nb * epoch  # global iteration

                # Warmup
                if ni <= nw:
                    xi = [0, nw]
                    accumulate = max(1, int(np.interp(ni, xi, [1, nbs / batch_size]).round()))
                    for j, pg in enumerate(optimizer.param_groups):
                        # Bias/BN groups warm up from warmup_bias_lr; others from 0.
                        # Legacy single-LR layout puts biases at group index 2; the
                        # differential layout tags each group with "is_bias".
                        is_bias_group = pg.get("is_bias", j == 2)
                        pg["lr"] = np.interp(ni, xi, [
                            args.get("warmup_bias_lr", 0.1) if is_bias_group else 0.0,
                            pg["initial_lr"] * lf(epoch)
                        ])
                        if "momentum" in pg:
                            pg["momentum"] = np.interp(ni, xi, [
                                args.get("warmup_momentum", 0.8), args["momentum"]
                            ])

                # Move batch to device
                batch["img"] = batch["img"].to(self.device, non_blocking=True)
                batch["bboxes"] = batch["bboxes"].to(self.device)
                batch["cls"] = batch["cls"].to(self.device)
                batch["batch_idx"] = batch["batch_idx"].to(self.device)

                # Forward
                with autocast(enabled=args.get("amp", True) and self.device.type == "cuda"):
                    preds = model(batch["img"])
                    loss, loss_items = criterion(preds, batch)

                # Backward
                scaler.scale(loss).backward()

                # Optimizer step
                if ni - last_opt_step >= accumulate:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    ema.update(model)
                    last_opt_step = ni

                mloss = (mloss * i + loss_items) / (i + 1)
                pbar.set_postfix({
                    "box": f"{mloss[0]:.4f}",
                    "cls": f"{mloss[1]:.4f}",
                    "dfl": f"{mloss[2]:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                })

            scheduler.step()

            # Validation
            val_dataset = build_dataset(args, split="val", augment=False)
            val_loader, _ = build_dataloader(
                val_dataset,
                batch_size=batch_size * 2,
                workers=args.get("workers", 8),
                augment=False,
                shuffle=False,
            )
            validator = Validator(
                model=ema.ema,
                dataloader=val_loader,
                device=self.device,
                conf=args.get("conf", 0.001),
                iou=args.get("iou", 0.7),
                max_det=args.get("max_det", 300),
            )
            metrics = validator.run()
            fitness = metrics.get("mAP50_95", 0.0)

            logger.info(
                f"Epoch {epoch+1}: "
                f"P={metrics.get('precision', 0):.4f}  R={metrics.get('recall', 0):.4f}  "
                f"mAP@50={metrics.get('mAP50', 0):.4f}  "
                f"mAP@50:95={metrics.get('mAP50_95', 0):.4f}"
            )

            box_l, cls_l, dfl_l = mloss[0].item(), mloss[1].item(), mloss[2].item()
            total_l = box_l + cls_l + dfl_l
            with open(results_csv, "a") as f:
                f.write(f"{epoch+1},{box_l:.6f},{cls_l:.6f},{dfl_l:.6f},{total_l:.6f},"
                        f"{metrics.get('precision', 0):.6f},{metrics.get('recall', 0):.6f},"
                        f"{metrics.get('mAP50', 0):.6f},{metrics.get('mAP50_95', 0):.6f}\n")

            # Checkpoint — mirrors Ultralytics layout (epoch, best_fitness, model, ema, updates, optimizer, ...)
            ckpt = {
                "epoch": epoch,
                "best_fitness": self.best_fitness,
                "model": deepcopy(model).half().state_dict(),
                "ema": deepcopy(ema.ema).half().state_dict(),
                "updates": ema.updates,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "train_args": args,
                "train_metrics": metrics,
            }
            torch.save(ckpt, last)

            if fitness >= self.best_fitness:
                self.best_fitness = fitness
                torch.save(ckpt, best)

            save_period = args.get("save_period", -1)
            if save_period > 0 and (epoch + 1) % save_period == 0:
                torch.save(ckpt, self.weights_dir / f"epoch{epoch+1}.pt")

            if stopper(epoch, fitness) or is_final:
                break

        logger.info(f"Training complete. Best mAP50:95={self.best_fitness:.4f}  → {best}")
        return best
