"""Experiment runner: trains one Mask R-CNN on all parts, then tests it part by part,
driven entirely by a YAML experiment config (see configs/experiments/).

Usage:
    python main.py --config configs/experiments/maskrcnn_adam3d.yaml
    python main.py --config configs/experiments/maskrcnn_adam3d.yaml --parts 1m1 tapa3m1
    python main.py --config configs/experiments/maskrcnn_adam3d.yaml --ckpt-path results/.../best.ckpt

Follows SuperDefectExperiments/main.py (``run_multi_class``): same config shape,
same ``metrics.csv`` / ``per_image_metrics.csv`` outputs, same W&B run layout.
"""

import argparse
import copy
import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from anomalib.engine import Engine
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from maskrcnn_modules.shared.reporting import compute_per_image_metrics

UNIFIED = "unified"


def build_wandb_logger(config: dict[str, Any], job_type: str) -> Any:
    """Build the run's WandbLogger from the experiment config's ``logging.wandb`` block."""
    from lightning.pytorch.loggers import WandbLogger

    wandb_args = dict(config.get("logging", {}).get("wandb", {}))
    group = wandb_args.pop("group", config["experiment"]["name"])
    return WandbLogger(
        name=f"{group}-{UNIFIED}",
        group=group,
        job_type=job_type,
        tags=[UNIFIED, config["experiment"].get("supervision_regime", "fully_supervised")],
        config=config,
        save_dir=_wandb_dir(config),
        reinit=True,
        **wandb_args,
    )


def _wandb_dir(config: dict[str, Any]) -> str:
    """Keep W&B's local run files beside the results instead of in the working directory."""
    results_dir = Path(config.get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir)


def _is_class_path_spec(value: Any) -> bool:
    return isinstance(value, dict) and "class_path" in value


def _resolve_init_args(value: Any) -> Any:
    """Recursively instantiate any nested ``{class_path, init_args}`` mapping.

    Lets a model's ``init_args`` (e.g. ``post_processor:``) name a nested
    component by class_path, the same way the top-level ``model:``/``data:``
    blocks do.
    """
    if _is_class_path_spec(value):
        return instantiate(value)
    if isinstance(value, list):
        return [_resolve_init_args(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_init_args(item) for key, item in value.items()}
    return value


def instantiate(spec: dict[str, Any]) -> Any:
    """Build an object from a ``{class_path, init_args}`` mapping."""
    module_name, _, class_name = spec["class_path"].rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    init_args = {key: _resolve_init_args(value) for key, value in spec.get("init_args", {}).items()}
    return cls(**init_args)


def build_engine(config: dict[str, Any], logger: Any, checkpoint: ModelCheckpoint) -> Engine:
    """Build an anomalib Engine with the config's trainer args and our callbacks."""
    callbacks = [checkpoint]
    if logger:
        # LearningRateMonitor refuses to run without a logger.
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return Engine(
        default_root_dir=config.get("results_dir", "./results"),
        logger=logger,
        callbacks=callbacks,
        **config.get("trainer", {}),
    )


def discover_parts(data_spec: dict[str, Any]) -> list[str]:
    """List the parts that have annotated defect images in the test annotation file."""
    with Path(data_spec["init_args"]["test_ann_file"]).open() as ann_file:
        coco = json.load(ann_file)
    annotated = {ann["image_id"] for ann in coco["annotations"]}
    return sorted({str(img.get("part", "unknown")) for img in coco["images"] if img["id"] in annotated})


def _save_per_image_metrics(engine: Engine, model: Any, datamodule: Any, output_dir: Path, part: str) -> None:
    """Run an extra predict pass and write one per-image metrics row per test image.

    ``metrics.csv`` only ever holds aggregated values, so this is the one
    place per-image scores are captured for bootstrap CIs.
    """
    predictions = engine.predict(model=model, datamodule=datamodule)
    rows = compute_per_image_metrics(predictions or [])
    part_dir = output_dir / part
    part_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(part_dir / "per_image_metrics.csv", index=False)


def _test(engine: Engine, model: Any, datamodule: Any) -> dict[str, Any]:
    test_results = engine.test(model=model, datamodule=datamodule)
    return dict(test_results[0]) if test_results else {}


def run_experiment(
    config: dict[str, Any],
    parts: list[str] | None = None,
    ckpt_path: Path | None = None,
) -> pd.DataFrame:
    """Train (unless ``ckpt_path`` is given), fit thresholds on validation, then test each part.

    One model and one pair of thresholds serve every part. The ``mean`` row
    averages the per-part rows, as in SuperDefectExperiments. The optional
    ``pooled`` row scores all test images in a single pass.
    """
    experiment = config["experiment"]
    seed_everything(experiment.get("seed", 42))

    output_dir = Path(config.get("results_dir", "./results")) / experiment["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_output_dir = output_dir if experiment.get("save_per_image_metrics") else None

    model = instantiate(copy.deepcopy(config["model"]))
    datamodule = instantiate(copy.deepcopy(config["data"]))

    job_type = "train" if ckpt_path is None else "eval"
    logger = build_wandb_logger(config, job_type) if config.get("logging", {}).get("wandb") else False
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "weights",
        filename="best",
        monitor="val/segm_mAP",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
        # A re-run replaces best.ckpt / last.ckpt (as it replaces metrics.csv)
        # instead of adding best-v1.ckpt beside the previous run's files.
        enable_version_counter=False,
    )
    engine = build_engine(config, logger, checkpoint)

    if ckpt_path is None:
        engine.fit(model=model, datamodule=datamodule)
        ckpt_path = Path(checkpoint.best_model_path) if checkpoint.best_model_path else None
    # The thresholds must belong to the weights being tested: this loads the
    # chosen checkpoint into ``model`` and refits image/pixel thresholds on
    # the pooled validation split, so the per-part tests below share them.
    engine.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    rows: list[dict[str, Any]] = []
    for part in parts or experiment.get("parts") or discover_parts(config["data"]):
        part_spec = copy.deepcopy(config["data"])
        part_spec["init_args"]["test_parts"] = [part]
        part_datamodule = instantiate(part_spec)

        rows.append({"category": part, **_test(engine, model, part_datamodule)})
        if per_image_output_dir is not None:
            _save_per_image_metrics(engine, model, part_datamodule, per_image_output_dir, part)

    frame = pd.DataFrame(rows).set_index("category")
    mean_row = frame.mean(numeric_only=True)
    mean_row.name = "mean"
    frame = pd.concat([frame, mean_row.to_frame().T])

    if experiment.get("test_pooled"):
        pooled_row = pd.Series(_test(engine, model, datamodule), name="pooled")
        frame = pd.concat([frame, pooled_row.to_frame().T])

    frame.index.name = "category"
    frame.to_csv(output_dir / "metrics.csv")
    with (output_dir / "resolved_config.yaml").open("w") as resolved_config_file:
        yaml.safe_dump(config, resolved_config_file, sort_keys=False)

    if logger:
        # Each per-part test logged the same metric keys into the training
        # run, so its summary would otherwise show whichever part ran last.
        for metric, value in frame.loc["mean"].items():
            logger.experiment.summary[f"mean_{metric}"] = value
        logger.experiment.finish()
        log_summary_to_wandb(config, frame)

    return frame


def log_summary_to_wandb(config: dict[str, Any], frame: pd.DataFrame) -> None:
    """Log the per-part + mean metrics table to a standalone 'summary' W&B run."""
    import wandb

    wandb_args = dict(config.get("logging", {}).get("wandb", {}))
    wandb_args.pop("log_model", None)  # a WandbLogger option, not a wandb.init one
    group = wandb_args.pop("group", config["experiment"]["name"])
    run = wandb.init(
        name="summary", group=group, job_type="summary", reinit=True, dir=_wandb_dir(config), **wandb_args,
    )
    table = frame.reset_index().rename(columns={"index": "category"})
    run.log({"metrics_summary": wandb.Table(dataframe=table)})
    for metric, value in frame.loc["mean"].items():
        run.summary[f"mean_{metric}"] = value
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--parts",
        nargs="+",
        default=None,
        help="Test only these parts instead of every part in the test annotations.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Skip training and evaluate this checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as config_file:
        config = yaml.safe_load(config_file)

    frame = run_experiment(config, parts=args.parts, ckpt_path=args.ckpt_path)
    print(frame)


if __name__ == "__main__":
    main()
