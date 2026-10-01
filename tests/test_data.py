import json

import torch
from anomalib.data.utils import Split

from maskrcnn_modules.data.coco_instance import CocoInstanceDataModule, CocoInstanceDataset
from maskrcnn_modules.data.dataclasses import InstanceBatch


def test_add_good_images_is_stratified_and_disjoint(coco_dataset):
    splits = {}
    for split in ("val", "test"):
        with coco_dataset[f"{split}_ann_file"].open() as f:
            splits[split] = [img for img in json.load(f)["images"] if img.get("source") == "good"]

    # ratio 1.0: one good image per annotated defect image, per part (2 per part in the fixture).
    for good_images in splits.values():
        assert sorted(img["model_id"] for img in good_images) == ["1m1", "1m1", "tapa3m1", "tapa3m1"]
    val_files = {img["file_name"] for img in splits["val"]}
    test_files = {img["file_name"] for img in splits["test"]}
    assert not val_files & test_files


def test_unannotated_defect_image_is_dropped_and_good_images_are_normal(coco_dataset):
    dataset = CocoInstanceDataset(
        coco_dataset["test_ann_file"], root=coco_dataset["root"], good_root=coco_dataset["good_root"], split=Split.TEST,
    )

    assert not any("unannotated" in path for path in dataset.samples.image_path)
    assert list(dataset.samples.label_index).count(0) == 4
    assert list(dataset.samples.label_index).count(1) == 4


def test_item_carries_instances_and_binary_mask(coco_dataset):
    dataset = CocoInstanceDataset(coco_dataset["train_ann_file"], root=coco_dataset["root"], split=Split.TRAIN)
    item = dataset[0]

    assert item.image.shape == (3, 64, 80)
    assert item.gt_label.item() is True
    assert item.gt_boxes.tolist() == [[8.0, 8.0, 24.0, 20.0], [44.0, 30.0, 60.0, 42.0]]
    assert item.gt_classes.tolist() == [1, 2]
    assert item.gt_instance_masks.shape == (2, 64, 80)
    # The binary mask is the union of the instances.
    assert torch.equal(item.gt_mask.bool(), item.gt_instance_masks.bool().any(dim=0))


def _fixture_instance_masks():
    expected = torch.zeros(2, 64, 80, dtype=torch.bool)
    expected[0, 8:20, 8:24] = True
    expected[1, 30:42, 44:60] = True
    return expected


def test_png_mask_source_reads_original_masks(coco_dataset):
    dataset = CocoInstanceDataset(
        coco_dataset["train_ann_file"], root=coco_dataset["root"], split=Split.TRAIN, mask_source="png",
    )
    item = dataset[0]

    # Each annotation picks up the PNG it was converted from, in annotation order.
    assert torch.equal(item.gt_instance_masks.bool(), _fixture_instance_masks())
    assert torch.equal(item.gt_mask.bool(), _fixture_instance_masks().any(dim=0))


def test_polygon_mask_source_rasterises_smaller_than_the_png(coco_dataset):
    dataset = CocoInstanceDataset(
        coco_dataset["train_ann_file"], root=coco_dataset["root"], split=Split.TRAIN, mask_source="polygon",
    )
    polygon_masks = dataset[0].gt_instance_masks.bool()

    png_areas = _fixture_instance_masks().flatten(1).sum(dim=1)
    assert (polygon_masks.flatten(1).sum(dim=1) < png_areas).all()


def test_png_mask_source_falls_back_to_polygons_without_a_matching_png(coco_dataset):
    polygons = CocoInstanceDataset(
        coco_dataset["train_ann_file"], root=coco_dataset["root"], split=Split.TRAIN, mask_source="polygon",
    )
    no_pngs = CocoInstanceDataset(
        coco_dataset["train_ann_file"], root=coco_dataset["root"], split=Split.TRAIN, mask_dir="no_such_dir",
    )

    assert torch.equal(no_pngs[0].gt_instance_masks, polygons[0].gt_instance_masks)
    assert torch.equal(no_pngs[0].gt_mask, polygons[0].gt_mask)


def test_good_item_has_empty_instances_and_zero_mask(coco_dataset):
    dataset = CocoInstanceDataset(
        coco_dataset["test_ann_file"], root=coco_dataset["root"], good_root=coco_dataset["good_root"], split=Split.TEST,
    )
    good_index = list(dataset.samples.label_index).index(0)
    item = dataset[good_index]

    assert item.gt_label.item() is False
    assert item.gt_boxes.shape == (0, 4)
    assert item.gt_instance_masks.shape == (0, 64, 80)
    assert not item.gt_mask.any()


def test_parts_filter(coco_dataset):
    dataset = CocoInstanceDataset(
        coco_dataset["test_ann_file"],
        root=coco_dataset["root"],
        good_root=coco_dataset["good_root"],
        split=Split.TEST,
        parts=["Tapa3M1"],
    )

    assert set(dataset.samples.part) == {"tapa3m1"}
    assert len(dataset) == 4


def test_collate_keeps_instance_fields_as_lists(coco_dataset):
    dataset = CocoInstanceDataset(
        coco_dataset["test_ann_file"], root=coco_dataset["root"], good_root=coco_dataset["good_root"], split=Split.TEST,
    )
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

    # Train holds abnormal images; val/test keep both their defect and good images.
    assert datamodule.train_data.has_anomalous and not datamodule.train_data.has_normal
    for subset in (datamodule.val_data, datamodule.test_data):
        assert list(subset.samples.label_index).count(0) == 4
        assert list(subset.samples.label_index).count(1) == 4
    assert (datamodule.name, datamodule.category) == ("adam3d", "unified")


def test_train_augmentation_keeps_instances_consistent(datamodule_kwargs):
    torch.manual_seed(0)
    datamodule = CocoInstanceDataModule(**datamodule_kwargs)
    datamodule.setup()

    for batch in datamodule.train_dataloader():
        assert batch.image.shape[-2:] == (64, 64)
        for boxes, classes, masks in zip(batch.gt_boxes, batch.gt_classes, batch.gt_instance_masks, strict=True):
            assert len(boxes) == len(classes) == len(masks)
            assert masks.shape[-2:] == (64, 64)
            assert ((boxes[:, 2:] - boxes[:, :2]) >= 1).all()
            assert masks.flatten(1).any(dim=1).all()


def test_test_parts_only_filters_the_test_split(datamodule_kwargs):
    datamodule = CocoInstanceDataModule(**datamodule_kwargs, test_parts=["1m1"])
    datamodule.setup()

    assert set(datamodule.test_data.samples.part) == {"1m1"}
    assert set(datamodule.val_data.samples.part) == {"1m1", "tapa3m1"}
