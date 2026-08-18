import cv2

"""
Crop a person's bounding box with additional
padding around the person.

padding=0.15 means add 15% of the person's
width/height around the detected box.
"""

def crop_person(frame, box, padding=0.15):
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    person_width = x2 - x1
    person_height = y2 - y1

    pad_x = int(person_width * padding)
    pad_y = int(person_height * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    return frame[y1:y2, x1:x2]


class WorkerSafetyDetector:
    """
    Second-stage detector that runs on a cropped worker image.

    Current checkpoint:
        class 0 = phone
        class 1 = face

    Face detections are ignored.
    Future version may add helmet and vest classes.
    """

    def predict(self, crop):
        return {"phone": []}
    """
    def __init__(
        self,
        weights,
        threshold=0.5,
        variant="RFDETRSmall",
    ):
        import rfdetr
        import torch

        self.threshold = threshold

        model_class = getattr(
            rfdetr,
            variant
        )

        self.model = model_class(
            pretrain_weights=weights
        )

        if torch.cuda.is_available():
            self.model.inference(
                dtype=torch.float16
            )
        else:
            self.model.inference()

    def predict(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return {"phone": []}

        rgb = cv2.cvtColor(
            crop_bgr,
            cv2.COLOR_BGR2RGB
        )

        detections = self.model.predict(
            rgb,
            threshold=self.threshold
        )

        results = {
            "phone": []
        }

        for i in range(len(detections)):
            class_id = int(
                detections.class_id[i]
            )

            # FPI checkpoint:
            # 0 = phone
            # 1 = face
            if class_id != 0:
                continue

            results["phone"].append({
                "confidence": float(
                    detections.confidence[i]
                ),
                "box": tuple(
                    map(
                        float,
                        detections.xyxy[i]
                    )
                )
            })

        return results
        """