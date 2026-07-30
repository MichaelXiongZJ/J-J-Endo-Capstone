"""Detector wrappers.

Thin layer over RF-DETR so the pipeline can also run against a scripted stub.
The stub exists because every downstream stage (tracking, homography, driver
association, all four rules) can be verified BEFORE a trained checkpoint
exists — otherwise a labeling-week delay blocks all pipeline work. It is a test
fixture only; it never appears in reported metrics.

RF-DETR (Apache-2.0) is a hard project requirement. Do not swap in YOLO, whose
AGPL-3.0 licence is incompatible with the J&J handoff (context.md §7.1).

Two names drifted from the implementation guide (verified against rfdetr 1.9.0,
which requirements.txt pins):
  * `rfdetr.util.coco_classes` -> `rfdetr.assets.coco_classes`
  * `model.optimize_for_inference()` -> `model.inference()`
"""

import cv2
import numpy as np
import supervision as sv

# rfdetr (and torch) are imported lazily throughout this module so that
# StubDetector — and the tests that use it — stay fast and dependency-free.


def coco_class_names():
    """{class_id: name} for the 80 COCO classes. person=1, cell phone=77.

    Only relevant to the Phase-2 zero-shot baseline. After fine-tuning, class_id
    indexes YOUR dataset's categories, not COCO's 80 (§4.4).
    """
    from rfdetr.assets.coco_classes import COCO_CLASSES
    return COCO_CLASSES


def model_class_ids(dataset_dir='data/dataset', split='valid'):
    """{name: predicted class_id} for a model fine-tuned on this dataset.

    A fine-tuned RF-DETR does NOT predict your COCO category ids. It re-indexes
    them to 0-based contiguous positions in sorted-id order, so a dataset with
    categories {1: forklift, 2: person} yields predictions 0 = forklift,
    1 = person — everything shifted down by one.

    This bites harder than the §4.4 warning suggests. §4.4 tells you to read ids
    from the dataset JSON rather than from memory, and doing exactly that still
    gives the wrong answer at inference, because the shift happens inside the
    model. The failure is silent: the detector reports excellent mAP (it is
    right, in its own indexing) while the pipeline matches nothing and emits zero
    events.

    Verified empirically by IoU-matching predictions against ground truth over 79
    objects: predicted 0 -> forklift, predicted 1 -> person.
    """
    import json
    import os

    path = os.path.join(dataset_dir, split, '_annotations.coco.json')
    with open(path) as f:
        cats = json.load(f)['categories']
    return {c['name']: i for i, c in enumerate(sorted(cats, key=lambda c: c['id']))}


class RFDetrDetector:
    """Fine-tuned RF-DETR. Returns sv.Detections; class_id per your dataset (§4.4).

    `variant` selects model size. The guide specifies RFDETRBase, which works in
    1.9.0 but is itself deprecated and removed in 2.0.0 (successors:
    RFDETRNano/Small/Medium/Large). Changing it changes accuracy — re-measure
    mAP (§4.6) if you do.
    """

    def __init__(self, weights=None, threshold=0.5, resolution=None,
                 optimize=True, variant='RFDETRBase'):
        import rfdetr
        import torch

        kwargs = {}
        if weights:
            kwargs['pretrain_weights'] = weights
        if resolution:
            # Must be divisible by 56. Costs compute; helps small/distant people,
            # the usual complaint on warehouse ceiling cameras.
            kwargs['resolution'] = resolution
        self.model = getattr(rfdetr, variant)(**kwargs)
        if optimize:
            # Must come AFTER weights are loaded. fp16 is ~8x on Tensor Core
            # GPUs (T4, RTX 30xx).
            if torch.cuda.is_available():
                self.model.inference(dtype=torch.float16)
            else:
                self.model.inference()
        self.threshold = threshold

    def __call__(self, frame_bgr):
        # CRITICAL: OpenCV loads BGR, RF-DETR expects RGB. Omitting this does not
        # crash — it quietly degrades accuracy. The #1 silent bug in the project
        # (§3, context.md §8.1). rtmlib, by contrast, takes BGR directly.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.model.predict(rgb, threshold=self.threshold)


class StubDetector:
    """Replays scripted detections, keyed by processed-frame index.

    script: {frame_index: [(x1, y1, x2, y2, class_id, confidence), ...]}
    Frames absent from the script yield zero detections.
    """

    def __init__(self, script):
        self.script = script
        self.i = -1

    def __call__(self, frame_bgr):
        self.i += 1
        rows = self.script.get(self.i, [])
        if not rows:
            return sv.Detections.empty()
        arr = np.asarray(rows, dtype=np.float32)
        return sv.Detections(
            xyxy=arr[:, :4].astype(np.float32),
            class_id=arr[:, 4].astype(int),
            confidence=arr[:, 5].astype(np.float32),
        )
