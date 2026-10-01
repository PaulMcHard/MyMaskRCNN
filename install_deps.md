# Setup

Dependencies are declared in `pyproject.toml`. The package installs into the same environment as `SuperDefectExperiments` (`sdx`); the only extra dependency is `pycocotools`.

```bash
# PyTorch first, with the CUDA build your GPU needs (cu128 for Blackwell-family GPUs).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Then this package, with test and wandb extras.
pip install -e .[dev,wandb]
```

Verify:

```bash
pytest -q
```

Run (see `docs/` for details):

```bash
# Once: add good images to the validation and test annotation files.
python scripts/add_good_images.py adam3d --good-root <path to 3d-adam-full-masked>

# Set data.init_args.root and good_root in the config, then:
python main.py --config configs/experiments/maskrcnn_adam3d_smoke.yaml   # quick pipeline check
python main.py --config configs/experiments/maskrcnn_adam3d.yaml         # full run
```

The MMDetection scripts at the repository root (`train.py`, `train_tb.py`, `validate.py`, `test.py`) are the earlier baseline. They need `mmengine`, `mmcv` and `mmdet`, which are not installed by `pyproject.toml`; `verification.py` checks that stack.
