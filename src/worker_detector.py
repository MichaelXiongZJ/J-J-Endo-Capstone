import cv2


def crop_person(frame, box, padding=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    px, py = int((x2 - x1) * padding), int((y2 - y1) * padding)
    return frame[max(0, y1-py):min(h, y2+py), max(0, x1-px):min(w, x2+px)]


class WorkerSafetyDetector:
    def __init__(self, weights=None, threshold=0.5):
        self.weights, self.threshold = weights, threshold

    def predict(self, crop):
        return {'phone': [], 'helmet': [], 'vest': []}