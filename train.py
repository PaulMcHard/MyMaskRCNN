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
    from mmdet.registry import DATASETS
    from mmdet.datasets import CocoDataset
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


def create_config(args):
    """Create MMDetection config for MaskRCNN."""
    
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
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
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
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
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
    
    # Dataset config
    cfg.dataset_type = dataset_type
    cfg.data_root = data_root
    
    # Training pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackDetInputs')
    ]
    
    # Test pipeline
    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ]
    
    # Training dataloader
    cfg.train_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=2,
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
    
    # Validation dataloader
    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=2,
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
    
    # Evaluator
    cfg.val_evaluator = dict(
        type='CocoMetric',
        ann_file=str(Path(args.annotations_dir) / 'annotations_val.json'),
        metric=['bbox', 'segm'],
        format_only=False,
        backend_args=None)
    cfg.test_evaluator = cfg.val_evaluator
    
    # Optimizer
    cfg.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='SGD', lr=args.learning_rate, momentum=0.9, weight_decay=0.0001))
    
    # Use IterBasedTrainLoop
    cfg.train_cfg = dict(
        type='IterBasedTrainLoop',
        max_iters=args.max_iter,
        val_interval=args.eval_period)
    
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')
    
    # Learning rate schedule
    cfg.param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=0.001,
            by_epoch=False,
            begin=0,
            end=500),
        dict(
            type='MultiStepLR',
            begin=0,
            end=args.max_iter,
            by_epoch=False,
            milestones=[args.max_iter // 2, int(args.max_iter * 0.75)],
            gamma=0.1)
    ]
    
    # Hooks
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=args.checkpoint_period, max_keep_ckpts=3),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook'))
    
    # Environment
    cfg.env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl'))
    
    # Visualizer
    cfg.visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer')
    
    # Logging
    cfg.log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
    cfg.log_level = 'INFO'
    cfg.default_scope = 'mmdet'
    
    # Output directory
    cfg.work_dir = args.output_dir
    
    # Load pretrained weights
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
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (images per iteration)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Base learning rate')
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Maximum number of iterations')
    parser.add_argument('--checkpoint-period', type=int, default=100,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--eval-period', type=int, default=50,
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


if __name__ == '__main__':
    main()