"""Regression tests for the fine-tuned class-id mapping.

This guards a bug that cost a full evaluation run. A fine-tuned RF-DETR does not
predict the COCO category ids in your dataset — it re-indexes them to 0-based
contiguous positions. With categories {1: forklift, 2: person} it predicts
0 = forklift, 1 = person.

Nothing crashes when you get this wrong. The detector reports excellent mAP (it
is correct in its own indexing) and the pipeline simply matches no class and
emits zero events, which reads like a rule bug rather than a plumbing bug.

§4.4 warns "do not hardcode class ids, read them from the dataset JSON" — and
following that advice exactly still gives the wrong answer at inference, because
the shift happens inside the model. Hence these tests.
"""

import json

from src.detector import model_class_ids


def write_dataset(tmp_path, categories, split='valid'):
    d = tmp_path / split
    d.mkdir(parents=True, exist_ok=True)
    (d / '_annotations.coco.json').write_text(json.dumps({
        'images': [], 'annotations': [], 'categories': categories,
    }))
    return str(tmp_path)


def test_one_based_categories_shift_down_to_zero_based(tmp_path):
    """Our actual dataset: {1: forklift, 2: person} -> {forklift: 0, person: 1}.

    Verified empirically by IoU-matching predictions against ground truth over 79
    objects on the trained checkpoint.
    """
    root = write_dataset(tmp_path, [{'id': 1, 'name': 'forklift'},
                                    {'id': 2, 'name': 'person'}])
    assert model_class_ids(root) == {'forklift': 0, 'person': 1}


def test_mapping_follows_sorted_id_order_not_file_order(tmp_path):
    """Category order inside the JSON must not change the mapping."""
    root = write_dataset(tmp_path, [{'id': 2, 'name': 'person'},
                                    {'id': 1, 'name': 'forklift'}])
    assert model_class_ids(root) == {'forklift': 0, 'person': 1}


def test_roboflow_dummy_category_shifts_everything(tmp_path):
    """Roboflow's COCO export sometimes inserts a dummy category at index 0
    (context.md §8.2). It is a real trained class, so it occupies position 0 and
    pushes the real classes up — exactly the shift §4.4 warns about, one layer
    further in.
    """
    root = write_dataset(tmp_path, [{'id': 0, 'name': 'objects'},
                                    {'id': 1, 'name': 'forklift'},
                                    {'id': 2, 'name': 'person'}])
    assert model_class_ids(root) == {'objects': 0, 'forklift': 1, 'person': 2}


def test_non_contiguous_ids_are_compacted(tmp_path):
    """Deleting a class in Roboflow can leave gaps in the id sequence."""
    root = write_dataset(tmp_path, [{'id': 3, 'name': 'forklift'},
                                    {'id': 7, 'name': 'person'}])
    assert model_class_ids(root) == {'forklift': 0, 'person': 1}


def test_reads_the_requested_split(tmp_path):
    write_dataset(tmp_path, [{'id': 1, 'name': 'forklift'},
                             {'id': 2, 'name': 'person'}], split='train')
    assert model_class_ids(str(tmp_path), split='train') == {'forklift': 0, 'person': 1}


def test_real_dataset_mapping_if_present():
    """Pin the mapping for the dataset actually on disk, when it exists.

    Skipped in a clean checkout; meaningful on a machine that has run the export.
    """
    import os
    import pytest

    if not os.path.exists('data/dataset/valid/_annotations.coco.json'):
        pytest.skip('no exported dataset on this machine')
    ids = model_class_ids('data/dataset')
    assert ids['forklift'] == 0 and ids['person'] == 1
