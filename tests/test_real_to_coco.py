"""Tests for the Roboflow -> COCO normaliser.

Two things must be right or real images silently poison the dataset:
the YOLO coordinate conversion, and the class-name mapping.
"""

import json
import os

import cv2
import numpy as np
import pytest

from scripts.real_to_coco import load_yolo_split, read_yolo_names


def build(tmp_path, names_line, label_lines, split='train', size=(480, 640)):
    root = tmp_path / 'ds'
    (root / split / 'images').mkdir(parents=True)
    (root / split / 'labels').mkdir(parents=True)
    h, w = size
    cv2.imwrite(str(root / split / 'images' / 'a.jpg'),
                np.full((h, w, 3), 60, np.uint8))
    (root / split / 'labels' / 'a.txt').write_text('\n'.join(label_lines))
    (root / 'data.yaml').write_text(names_line + '\nnc: 3\n')
    return str(root)


def test_reads_inline_yaml_names(tmp_path):
    root = build(tmp_path, "names: ['forklift', 'human', 'pallet']", [])
    assert read_yolo_names(root) == ['forklift', 'human', 'pallet']


def test_reads_block_yaml_names(tmp_path):
    root = build(tmp_path, 'names:\n  - forklift\n  - human', [])
    assert read_yolo_names(root) == ['forklift', 'human']


def test_yolo_boxes_convert_to_absolute_coco_xywh(tmp_path):
    """YOLO is normalised centre+size; COCO is absolute top-left+size. Getting
    this wrong shifts every box by half its width and height, which still trains
    to a plausible-looking loss."""
    root = build(tmp_path, "names: ['forklift', 'human']",
                 ['0 0.5 0.5 0.25 0.5'], size=(480, 640))
    recs = load_yolo_split(root, 'train', ['forklift', 'human'], {})
    _p, _fn, w, h, anns = recs[0]
    assert (w, h) == (640, 480)
    cls, bbox = anns[0]
    assert cls == 'forklift'
    assert bbox == [240.0, 120.0, 160.0, 240.0]


def test_class_names_are_normalised_onto_ours(tmp_path):
    """'human' means person. Datasets disagree on wording; we must not."""
    root = build(tmp_path, "names: ['forklift', 'human']",
                 ['0 0.5 0.5 0.2 0.2', '1 0.3 0.3 0.1 0.2'])
    recs = load_yolo_split(root, 'train', ['forklift', 'human'], {})
    assert sorted(c for c, _ in recs[0][4]) == ['forklift', 'person']


def test_unmapped_classes_are_dropped_and_reported(tmp_path):
    """A class we cannot confidently map must NOT be folded into person or
    forklift — a mislabeled box is worse than a missing one."""
    root = build(tmp_path, "names: ['forklift', 'pallet']",
                 ['0 0.5 0.5 0.2 0.2', '1 0.3 0.3 0.1 0.2'])
    unmapped = {}
    recs = load_yolo_split(root, 'train', ['forklift', 'pallet'], unmapped)
    assert [c for c, _ in recs[0][4]] == ['forklift']
    assert unmapped == {'pallet': 1}


def test_image_without_label_file_is_kept_as_background(tmp_path):
    """Negative images are legitimate training signal, not an error."""
    root = build(tmp_path, "names: ['forklift']", [])
    os.remove(os.path.join(root, 'train', 'labels', 'a.txt'))
    recs = load_yolo_split(root, 'train', ['forklift'], {})
    assert len(recs) == 1 and recs[0][4] == []


def test_missing_split_returns_empty(tmp_path):
    root = build(tmp_path, "names: ['forklift']", [])
    assert load_yolo_split(root, 'test', ['forklift'], {}) == []
