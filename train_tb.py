import os
import argparse
from pathlib import Path
import json
import sys
import torch

# Check imports
try:
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.registry import DATASETS, HOOKS
    from mmdet.datasets import CocoDataset
    from mmengine.hooks import Hook
except ImportError as e:
    print(f"Error importing MMDetection: {e}")
    sys.exit(1)

@DATASETS.register_module()
class Adam3DDefectDataset(CocoDataset):
    """Custom dataset for 3D-ADAM defect detection."""
    
    METAINFO = {
        'classes': ('bulge', 'crack', 'hole', 'under_extrusion', 
                   'over_extrusion', 'cut', 'scratch'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                   (106, 0, 228), (0, 60, 100), (0, 80, 100)]
    }

@HOOKS.register_module()
class ValLossHook(Hook):
    """Hook to compute and log validation loss to TensorBoard."""
    
    def __init__(self, interval=1):
        self.interval = interval
        self._val_loss_dataloader = None
    
    def _build_val_loss_dataloader(self, runner):
        """Build a separate dataloader for validation loss computation."""
        if self._val_loss_dataloader is not None:
            return self._val_loss_dataloader
        
        from mmengine.dataset import DefaultSampler
        from mmengine.registry import DATASETS
        from torch.utils.data import DataLoader
        
        # Get validation dataset config
        val_dataset_cfg = runner.cfg.val_dataloader.dataset.copy()
        
        # Change to training mode pipeline for loss computation
        val_dataset_cfg['test_mode'] = False
        val_dataset_cfg['pipeline'] = runner.cfg.train_dataloader.dataset.pipeline
        
        # Build dataset using registry
        val_dataset = DATASETS.build(val_dataset_cfg)
        
        # Create dataloader
        sampler = DefaultSampler(val_dataset, shuffle=False)
        self._val_loss_dataloader = DataLoader(
            val_dataset,
            batch_size=runner.cfg.val_dataloader.batch_size,
            sampler=sampler,
            num_workers=runner.cfg.val_dataloader.num_workers,
            collate_fn=runner.val_dataloader.collate_fn,
            persistent_workers=runner.cfg.val_dataloader.get('persistent_workers', False),
        )
        
        return self._val_loss_dataloader
    
    def _parse_loss_value(self, value):
        """Parse loss value that could be tensor, list, or scalar."""
        if torch.is_tensor(value):
            return value.item()
        elif isinstance(value, list):
            # If it's a list of tensors, sum them
            return sum(v.item() if torch.is_tensor(v) else v for v in value)
        else:
            return value
    
    def after_val_epoch(self, runner, metrics=None):
        """Compute validation loss after each validation epoch."""
        if runner.epoch % self.interval != 0:
            return
        
        # Build validation loss dataloader
        try:
            val_loss_dataloader = self._build_val_loss_dataloader(runner)
        except Exception as e:
            runner.logger.error(f'Failed to build validation loss dataloader: {e}')
            return
        
        # Set model to train mode to get losses (but don't update weights)
        mode = runner.model.training
        runner.model.train()
        
        # Compute losses on validation set
        total_loss = 0
        loss_dict_sum = {}
        num_samples = 0
        num_errors = 0
        
        for idx, data_batch in enumerate(val_loss_dataloader):
            try:
                with torch.no_grad():
                    # Move data to device
                    data = runner.model.data_preprocessor(data_batch, training=True)
                    
                    # Unpack the preprocessed data
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    # Forward pass - pass unpacked arguments
                    loss_dict = runner.model.loss(inputs, data_samples)
                    
                    # Accumulate losses
                    for key, value in loss_dict.items():
                        parsed_value = self._parse_loss_value(value)
                        
                        if key not in loss_dict_sum:
                            loss_dict_sum[key] = 0.0
                        
                        loss_dict_sum[key] += parsed_value
                    
                    # Calculate total loss (sum all loss components)
                    loss_total = sum(self._parse_loss_value(v) 
                                   for k, v in loss_dict.items() 
                                   if 'loss' in k)
                    total_loss += loss_total
                    
                    num_samples += 1
                    
            except Exception as e:
                num_errors += 1
                if num_errors <= 3:  # Only log first 3 errors
                    runner.logger.warning(f'Error computing loss for val sample {idx}: {str(e)}')
                    import traceback
                    if num_errors == 1:  # Show full trace for first error
                        runner.logger.debug(traceback.format_exc())
                continue
        
        # Restore original model mode
        runner.model.train(mode)
        
        if num_samples == 0:
            runner.logger.warning('No valid validation samples for loss computation')
            return
        
        if num_errors > 0:
            runner.logger.warning(f'Skipped {num_errors}/{num_samples + num_errors} validation samples due to errors')
        
        # Calculate average losses
        avg_total_loss = total_loss / num_samples
        avg_losses = {k: v / num_samples for k, v in loss_dict_sum.items()}
        
        # Log to TensorBoard via visualizer
        if runner.visualizer is not None:
            runner.visualizer.add_scalar(
                'val/total_loss', 
                avg_total_loss, 
                runner.iter
            )
            for loss_name, loss_value in avg_losses.items():
                runner.visualizer.add_scalar(
                    f'val/{loss_name}',
                    loss_value,
                    runner.iter
                )
        
        # Also log to console
        runner.logger.info(f'Validation Total Loss: {avg_total_loss:.4f} (computed on {num_samples} samples)')
        for loss_name, loss_value in avg_losses.items():
            runner.logger.info(f'  val/{loss_name}: {loss_value:.4f}')

def create_config(args):
    """Create MMDetection config for MaskRCNN with regularization."""
    
    from mmengine.config import Config
    cfg = Config()
    
    # Dataset settings
    dataset_type = 'Adam3DDefectDataset'
    data_root = args.dataset_root
    
    # Get class names
    with open(Path(args.annotations_dir) / 'annotations_train.json', 'r') as f:
        coco_data = json.load(f)
    num_classes = len(coco_data['categories'])
    
    print(f"\nDataset configuration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {[cat['name'] for cat in coco_data['categories']]}")
    
    # Model configuration
    cfg.model = dict(
        type='MaskRCNN',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=True,
            pad_size_divisor=32),
        #backbone=dict(
        #    type='ResNet',
        #    depth=50,
        #    num_stages=4,
        #    out_indices=(0, 1, 2, 3),
        #    frozen_stages=1,
        #    norm_cfg=dict(type='BN', requires_grad=True),
        #    norm_eval=True,
        #    style='pytorch',
        #    init_cfg=None),
        backbone=dict(
            type='ResNeXt',
            depth=101,
            groups=64,  # More feature groups
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            style='pytorch',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnext101_64x4d')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            #anchor_generator=dict(
            #    type='AnchorGenerator',
            #    scales=[8],
            #    ratios=[0.5, 1.0, 2.0],
            #    strides=[4, 8, 16, 32, 64]),
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[2, 4, 8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=False, 
                    loss_weight=1.0),  # Removed label_smoothing
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(
                    type='CrossEntropyLoss', 
                    use_mask=True, 
                    loss_weight=1.0))),  # Removed label_smoothing
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5)))
    
    cfg.dataset_type = dataset_type
    cfg.data_root = data_root
    
    # Enhanced training pipeline with augmentation
    train_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),

        # Multi-scale training (critical for generalization)
        dict(
            type='RandomResize',
            scale=[ (1280, 1280), (1792, 1792)],
            keep_ratio=True
        ),

        # Random crops (forces model to detect partial defects)
        dict(
            type='RandomCrop',
            crop_size=(1024, 1024),
            crop_type='absolute',
            allow_negative_crop=True
        ),

        # Geometric augmentations
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),  # Increased from 0.3
    
        dict(type='PackDetInputs')
    ]
    
    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ]
    
    # Training dataloader
    cfg.train_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=8,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        batch_sampler=dict(type='AspectRatioBatchSampler'),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=str(Path(args.annotations_dir) / 'annotations_train.json'),
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline))
    
    cfg.val_dataloader = dict(
        batch_size=4,
        num_workers=8,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=str(Path(args.annotations_dir) / 'annotations_val.json'),
            data_prefix=dict(img=''),
            test_mode=True,
            pipeline=test_pipeline))
    
    cfg.test_dataloader = cfg.val_dataloader
    
    cfg.val_evaluator = dict(
        type='CocoMetric',
        ann_file=str(Path(args.annotations_dir) / 'annotations_val.json'),
        metric=['bbox', 'segm'],
        format_only=False,
        backend_args=None)
    cfg.test_evaluator = cfg.val_evaluator
    
    # Optimizer with stronger regularization
    cfg.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(
            type='SGD', 
            lr=args.learning_rate, 
            momentum=0.9, 
            weight_decay=0.001  # Increased from 0.0001
        ),
        clip_grad=dict(max_norm=35, norm_type=2)  # Gradient clipping
    )
    
    cfg.train_cfg = dict(
        type='IterBasedTrainLoop',
        max_iters=args.max_iter,
        val_interval=args.eval_period)
    
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')
    
    warmup_iters = min(500, args.max_iter // 20)  # 5% warmup

    # Cosine annealing learning rate
    cfg.param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=0.001,
            by_epoch=False,
            begin=0,
            end=warmup_iters),
        dict(
            type='CosineAnnealingLR',
            T_max=args.max_iter - 500,
            eta_min=args.learning_rate * 0.01,
            begin=500,
            end=args.max_iter,
            by_epoch=False)
    ]
    
    # Custom hooks
    cfg.custom_hooks = [
        dict(type='ValLossHook', interval=1, priority='LOW')
    ]
    
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook', 
            interval=args.checkpoint_period, 
            max_keep_ckpts=3,
            save_best='coco/segm_mAP',  # Save best model based on segmentation mAP
            rule='greater'  # Higher is better
        ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook'))
    
    cfg.env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl'))
    
    cfg.visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=[
            dict(type='LocalVisBackend'),
            dict(type='TensorboardVisBackend')
        ],
        name='visualizer')
    
    cfg.log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
    cfg.log_level = 'INFO'
    cfg.default_scope = 'mmdet'
    cfg.work_dir = args.output_dir
    cfg.load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
    cfg.resume = False
    
    return cfg

def main():
    parser = argparse.ArgumentParser(description='Train MaskRCNN on 3D-ADAM dataset using MMDetection')
    
    # Dataset paths
    parser.add_argument('--dataset-root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--annotations-dir', type=str, required=True,
                        help='Directory containing COCO JSON annotations')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (images per iteration)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Base learning rate')
    parser.add_argument('--max-iter', type=int, default=20000,
                        help='Maximum number of iterations')
    parser.add_argument('--checkpoint-period', type=int, default=1000,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--eval-period', type=int, default=500,
                        help='Evaluate every N iterations')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for models and logs')
    
    # Resume
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Will train on GPU")
    else:
        print(f"Will train on CPU (slower)")
    
    # Create config
    print("\nCreating configuration...")
    cfg = create_config(args)
    
    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Save config
    cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
    print(f"Configuration saved to: {os.path.join(cfg.work_dir, 'config.py')}")
    
    # Print TensorBoard info
    tensorboard_dir = os.path.join(cfg.work_dir, 'vis_data')
    print(f"\n{'='*80}")
    print(f"TensorBoard Visualization Enabled (Including Validation Loss)")
    print(f"{'='*80}")
    print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    print(f"\nTo view training progress, run in a separate terminal:")
    print(f"  tensorboard --logdir={tensorboard_dir}")
    print(f"\nThen open your browser to: http://localhost:6006")
    print(f"\nYou will see:")
    print(f"  - Training losses (train/loss, train/loss_rpn_cls, etc.)")
    print(f"  - Validation losses (val/total_loss, val/loss_rpn_cls, etc.)")
    print(f"  - Validation metrics (coco/bbox_mAP, coco/segm_mAP, etc.)")
    print(f"{'='*80}\n")
    
    # Build runner
    print("\nBuilding runner...")
    runner = Runner.from_cfg(cfg)
    
    # Start training
    print(f"\nStarting training...")
    print(f"Output directory: {cfg.work_dir}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Batch size: {args.batch_size}")
    
    if args.resume:
        runner.resume()
    else:
        runner.train()
    
    print("\nTraining completed!")
    print(f"\nView training results:")
    print(f"  tensorboard --logdir={tensorboard_dir}")


if __name__ == '__main__':
    main()