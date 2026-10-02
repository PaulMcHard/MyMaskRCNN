import pytest
import torch
from anomalib.data.utils import Split

from maskrcnn_modules.data.coco_instance import CocoInstanceDataModule, CocoInstanceDataset, segmentation_to_mask
from maskrcnn_modules.data.dataclasses import InstanceBatch
from tests.conftest import FIRST_INSTANCE, SECOND_INSTANCE, rectangle_mask


def _dataset(coco_dataset, split, **kwargs):
    ann_file = coco_dataset[f"{Split(split).value}_ann_file"]
    return CocoInstanceDataset(ann_file, root=coco_dataset["root"], split=split, **kwargs)


def _instance_masks():
    return torch.from_numpy(rectangle_mask(*FIRST_INSTANCE)), torch.from_numpy(rectangle_mask(*SECOND_INSTANCE))


def test_good_images_are_normal_and_defect_images_abnormal(coco_dataset):
    dataset = _dataset(coco_dataset, Split.TEST)

    assert list(dataset.samples.label_index).count(0) == 4
    assert list(dataset.samples.label_index).count(1) == 4


def test_item_carries_exact_instances_and_their_union(coco_dataset):
    dataset = _dataset(coco_dataset, Split.TRAIN)
    item = dataset[0]
    first, second = _instance_masks()

    assert item.image.shape == (3, 64, 80)
    assert item.gt_label.item() is True
    assert item.gt_boxes.tolist() == [[8.0, 8.0, 24.0, 20.0], [44.0, 30.0, 60.0, 42.0]]
    assert sorted(dataset.categories[c] for c in item.gt_classes.tolist()) == ["bulge", "burr"]
    assert torch.equal(item.gt_instance_masks.bool(), torch.stack([first, second]))
    assert torch.equal(item.gt_mask.bool(), first | second)


def test_good_item_has_empty_instances_and_zero_mask(coco_dataset):
    dataset = _dataset(coco_dataset, Split.TEST)
    item = dataset[list(dataset.samples.label_index).index(0)]

    assert item.gt_label.item() is False
    assert item.gt_boxes.shape == (0, 4)
    assert item.gt_instance_masks.shape == (0, 64, 80)
    assert not item.gt_mask.any()


def test_ignore_classes_become_background(coco_dataset):
    dataset = _dataset(coco_dataset, Split.TRAIN, ignore_classes=["burr"])
    item = dataset[0]
    first, _ = _instance_masks()

    assert [dataset.categories[c] for c in item.gt_classes.tolist()] == ["bulge"]
    assert torch.equal(item.gt_mask.bool(), first)


def test_defect_images_left_without_instances_are_dropped(coco_dataset):
    # The UnderExtrusion test images hold only under_extrusion and burr instances.
    dataset = _dataset(coco_dataset, Split.TEST, ignore_classes=["under_extrusion", "burr"])

    assert not any("UnderExtrusion" in path for path in dataset.samples.image_path)
    assert list(dataset.samples.label_index).count(1) == 2


def test_unknown_ignore_class_is_an_error(coco_dataset):
    with pytest.raises(ValueError, match="not categories"):
        _dataset(coco_dataset, Split.TRAIN, ignore_classes=["wobble"])


def test_polygon_segmentation_still_decodes():
    mask = segmentation_to_mask([[2, 2, 9, 2, 9, 6, 2, 6]], 10, 12)

    assert mask.shape == (10, 12)
    assert mask[3:6, 3:9].all()
    assert not mask[8:].any()


def test_parts_filter_is_case_insensitive(coco_dataset):
    dataset = _dataset(coco_dataset, Split.TEST, parts=["tapa3m1"])

    assert set(dataset.samples.part) == {"Tapa3M1"}
    assert len(dataset) == 4


def test_collate_keeps_instance_fields_as_lists(coco_dataset):
    dataset = _dataset(coco_dataset, Split.TEST)
    good_index = list(dataset.samples.label_index).index(0)
    defect_index = list(dataset.samples.label_index).index(1)
    batch = dataset.collate_fn([dataset[defect_index], dataset[good_index]])

    assert isinstance(batch, InstanceBatch)
    assert batch.image.shape == (2, 3, 64, 80)
    assert batch.gt_mask.shape == (2, 64, 80)
    assert [len(boxes) for boxes in batch.gt_boxes] == [2, 0]
    assert [target["masks"].shape[0] for target in batch.targets()] == [2, 0]


def test_datamodule_keeps_explicit_splits(datamodule_kwargs):
    datamodule = CocoInstanceDataModule(**datamodule_kwargs)
    datamodule.setup()

    # Train holds abnormal images only; val/test keep both their defect and good images.
    assert datamodule.train_data.has_anomalous and not datamodule.train_data.has_normal
    for subset in (datamodule.val_data, datamodule.test_data):
        assert list(subset.samples.label_index).count(0) == 4
        assert list(subset.samples.label_index).count(1) == 4
    assert (datamodule.name, datamodule.category) == ("adam3d", "unified")


def test_datamodule_applies_ignore_classes_to_every_split(datamodule_kwargs):
    datamodule = CocoInstanceDataModule(**datamodule_kwargs, ignore_classes=["burr"])
    datamodule.setup()

    for subset in (datamodule.train_data, datamodule.val_data, datamodule.test_data):
        defect_index = list(subset.samples.label_index).index(1)
        assert "burr" not in {subset.categories[c] for c in subset[defect_index].gt_classes.tolist()}


def test_train_augmentation_keeps_instances_consistent(datamodule_kwargs):
    torch.manual_seed(0)
    datamodule = CocoInstanceDataModule(**datamodule_kwargs)
    datamodule.setup()

    for batch in datamodule.train_dataloader():
        assert batch.image.shape[-2:] == (64, 64)
        for boxes, classes, masks, gt_mask in zip(
            batch.gt_boxes, batch.gt_classes, batch.gt_instance_masks, batch.gt_mask, strict=True,
        ):
            assert len(boxes) == len(classes) == len(masks)
            assert masks.shape[-2:] == (64, 64)
            assert ((boxes[:, 2:] - boxes[:, :2]) >= 1).all()
            assert masks.flatten(1).any(dim=1).all()
            # The binary mask follows the same crop and flips as the instances.
            union = masks.bool().any(dim=0) if len(masks) else torch.zeros_like(gt_mask)
            assert torch.equal(gt_mask.bool(), union)


def test_dataloaders_keep_workers_alive(datamodule_kwargs):
    datamodule = CocoInstanceDataModule(**{**datamodule_kwargs, "num_workers": 2})
    datamodule.setup()

    for loader in (datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()):
        assert loader.persistent_workers
        assert loader.num_workers == 2


def test_test_parts_only_filters_the_test_split(datamodule_kwargs):
    datamodule = CocoInstanceDataModule(**datamodule_kwargs, test_parts=["1M1"])
    datamodule.setup()

    assert set(datamodule.test_data.samples.part) == {"1M1"}
    assert set(datamodule.val_data.samples.part) == {"1M1", "Tapa3M1"}
