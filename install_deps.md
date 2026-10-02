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
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print a +cu build and True
pytest -q
```

The configs set `trainer.accelerator: gpu`, so a CPU-only torch fails at start-up instead of silently training on the CPU.

Run (see `docs/` for details):

```bash
# The annotation files in adam3d/superdefect_default/ are committed. To rebuild them
# (needs the SuperDefect repository next to this one, and 3d-adam-full-masked):
python scripts/build_coco_annotations.py --superdefect-root ../SuperDefect --dataset-root <path to 3d-adam-full-masked>

# Set data.init_args.root in the config to your 3d-adam-full-masked path, then:
python main.py --config configs/experiments/maskrcnn_adam3d_smoke.yaml   # quick pipeline check
python main.py --config configs/experiments/maskrcnn_adam3d.yaml         # full run
```

The MMDetection scripts at the repository root (`train.py`, `train_tb.py`, `validate.py`, `test.py`) are the earlier baseline and use the older `adam3d/annotations*.json`. They need `mmengine`, `mmcv` and `mmdet`, which are not installed by `pyproject.toml`; `verification.py` checks that stack.
