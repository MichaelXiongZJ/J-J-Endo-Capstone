# Phone Detection Model

The RF-DETR checkpoint is intentionally excluded from Git because the file exceeds GitHub's size limit.

Place the checkpoint here before running inference:

```text
models/phone/checkpoint_best_total.pth
```

Current checkpoint:

- Dataset: FPI-Det
- Classes:
  - 0 = phone
  - 1 = face

Training subset:

- 1,500 training images
- 300 validation images

Best validation metrics:

- AP50 (phone): 0.93
- Recall (phone): 0.86
