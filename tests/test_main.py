import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def main_module():
    spec = importlib.util.spec_from_file_location("main", REPO_ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(datamodule_kwargs, results_dir):
    return {
        "experiment": {
            "name": "maskrcnn_test",
            "seed": 0,
            "save_per_image_metrics": True,
            "test_pooled": True,
        },
        "model": {
            "class_path": "maskrcnn_modules.maskrcnn.MaskRCNN",
            "init_args": {"pretrained": "none", "min_size": 64, "max_size": 80, "visualizer": False},
        },
        "data": {
            "class_path": "maskrcnn_modules.data.CocoInstanceDataModule",
            "init_args": {key: str(value) if isinstance(value, Path) else value for key, value in datamodule_kwargs.items()},
        },
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "max_steps": 2,
            "val_check_interval": 2,
            "check_val_every_n_epoch": None,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        "results_dir": str(results_dir),
    }


def test_instantiate_resolves_nested_class_paths(main_module):
    model = main_module.instantiate({
        "class_path": "maskrcnn_modules.maskrcnn.MaskRCNN",
        "init_args": {
            "pretrained": "none",
            "visualizer": False,
            "post_processor": {
                "class_path": "maskrcnn_modules.maskrcnn.DetectionPostProcessor",
                "init_args": {"enable_normalization": False},
            },
        },
    })

    assert model.post_processor.enable_normalization is False


def test_discover_parts_lists_parts_with_annotated_images(main_module, datamodule_kwargs):
    assert main_module.discover_parts({"init_args": datamodule_kwargs}) == ["1M1", "Tapa3M1"]


def test_run_experiment_writes_metrics_and_per_image_csvs(main_module, datamodule_kwargs, tmp_path):
    config = _config(datamodule_kwargs, tmp_path)

    frame = main_module.run_experiment(config)

    assert list(frame.index) == ["1M1", "Tapa3M1", "mean", "pooled"]
    for column in ("image_AUROC", "pixel_AUPRO", "pixel_BFScore1", "segm_mAP"):
        assert column in frame.columns

    output_dir = tmp_path / "maskrcnn_test"
    saved = pd.read_csv(output_dir / "metrics.csv", index_col="category")
    assert list(saved.index) == ["1M1", "Tapa3M1", "mean", "pooled"]
    with (output_dir / "resolved_config.yaml").open() as f:
        assert yaml.safe_load(f)["experiment"]["name"] == "maskrcnn_test"
    assert (output_dir / "weights" / "best.ckpt").exists()

    per_image = pd.read_csv(output_dir / "1M1" / "per_image_metrics.csv")
    # 2 defect + 2 good test images for this part; the image without masks was excluded by the generator.
    assert len(per_image) == 4
    assert {"image_path", "gt_label", "pred_label", "pred_score", "pixel_iou", "pixel_dice", "pixel_ap"} <= set(
        per_image.columns,
    )
    assert sorted(per_image.gt_label) == [0, 0, 1, 1]


def test_run_experiment_can_evaluate_an_existing_checkpoint(main_module, datamodule_kwargs, tmp_path):
    config = _config(datamodule_kwargs, tmp_path)
    main_module.run_experiment(config, parts=["1M1"])
    checkpoint = tmp_path / "maskrcnn_test" / "weights" / "best.ckpt"

    config["experiment"]["name"] = "maskrcnn_eval_only"
    config["experiment"]["test_pooled"] = False
    frame = main_module.run_experiment(config, parts=["Tapa3M1"], ckpt_path=checkpoint)

    assert list(frame.index) == ["Tapa3M1", "mean"]
    assert not (tmp_path / "maskrcnn_eval_only" / "weights" / "best.ckpt").exists()
