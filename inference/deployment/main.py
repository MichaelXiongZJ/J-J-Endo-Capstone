import os
import cv2
import threading
import time
import logging
import signal
import sys
from flask import Flask
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from rfdetr import RFDETRBase

RTSP_URL = os.environ.get("RTSP_URL", "rtsp://localhost:8554/demo")
STALE_THRESHOLD_SEC = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,   # overrides any prior config
)
logger = logging.getLogger("detector")

# --- Prometheus metrics ---
FRAME_COUNTER = Counter("frames_processed_total", "Total frames processed")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Inference latency in seconds")
DETECTION_COUNT = Counter("detections_total", "Total detections across all frames")


class LatestFrameReader:
    """Continuously reads frames in a background thread and always
    keeps only the most recent one, so inference never lags behind.
    Reconnects automatically if the stream drops."""

    def __init__(self, src):
        self.src = src
        self.lock = threading.Lock()
        self.frame = None
        self.last_frame_time = time.time()
        self.stopped = False
        self.cap = self._open()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open(self):
        logger.info("Attempting to connect to %s", self.src)
        cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)   # 5s connect timeout (OpenCV 4.6+)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream: {self.src}")
        logger.info("Successfully connected to %s", self.src)
        return cap

    def _update(self):
        fail_count = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                fail_count += 1
                logger.warning("Frame read failed (%d consecutive)", fail_count)
                if fail_count >= 30:  # ~a few seconds of failures -> reconnect
                    logger.warning("Reconnecting to %s", self.src)
                    self.cap.release()
                    time.sleep(1)
                    try:
                        self.cap = self._open()
                        fail_count = 0
                    except RuntimeError:
                        time.sleep(2)
                continue
            fail_count = 0
            with self.lock:
                self.frame = frame
                self.last_frame_time = time.time()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def seconds_since_last_frame(self):
        with self.lock:
            return time.time() - self.last_frame_time

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=5)
        self.cap.release()


# --- health + metrics endpoints for k8s probes and Prometheus ---
health_app = Flask(__name__)
reader_ref = {"reader": None}


@health_app.route("/healthz")
def healthz():
    reader = reader_ref["reader"]
    if reader is None:
        return "not ready", 503
    age = reader.seconds_since_last_frame()
    if age > STALE_THRESHOLD_SEC:
        return f"stale ({age:.1f}s since last frame)", 503
    return "ok", 200


@health_app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


def run_health_server():
    health_app.run(host="0.0.0.0", port=8080)


def main():
    show_window = "--show" in sys.argv  # opt-in only, off by default for containers

    model = RFDETRBase()
    reader = LatestFrameReader(RTSP_URL)
    reader_ref["reader"] = reader
    logger.info("Connected to RTSP stream")

    threading.Thread(target=run_health_server, daemon=True).start()

    shutdown = {"flag": False}

    def handle_sigterm(signum, frame):
        logger.info("Received shutdown signal, exiting cleanly")
        shutdown["flag"] = True

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    frame_count = 0
    last_log = time.time()

    try:
        while not shutdown["flag"]:
            frame = reader.read()
            if frame is None:
                time.sleep(0.005)
                continue

            start = time.time()
            detections = model.predict(frame)
            elapsed = time.time() - start
            inference_ms = elapsed * 1000

            frame_count += 1
            FRAME_COUNTER.inc()
            INFERENCE_LATENCY.observe(elapsed)
            DETECTION_COUNT.inc(len(detections.xyxy))

            if time.time() - last_log >= 5:  # log every 5 seconds
                # logger.info("Processed %d frames, last inference: %.1fms, %d detections",
                #             frame_count, inference_ms, len(detections.xyxy))
                last_log = time.time()

            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                confidence = float(detections.confidence[i])
                class_id = int(detections.class_id[i])
                label = f"{class_id}: {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if show_window:
                cv2.imshow("RF-DETR RTSP Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except Exception:
        logger.exception("Fatal error in main loop")
        raise
    finally:
        reader.stop()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()