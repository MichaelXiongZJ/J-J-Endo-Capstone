# RF-DETR RTSP Detection Service — Deployment Guide

This service connects to an RTSP camera stream, runs real-time object
detection with RF-DETR, and exposes health and Prometheus metrics endpoints.
It is designed to run as a container on Kubernetes.

## What's included

- `main.py` — the detection service (RTSP reader + RF-DETR inference + Flask
  health/metrics server)
- `requirements.txt` — Python dependencies
- `Dockerfile` — builds the container image
- `deployment.yaml` — Kubernetes Deployment + Service

## Prerequisites

- `docker` installed and logged in to a container registry
  (`docker login`)
- `kubectl` configured to point at the target Kubernetes cluster
  (`kubectl get nodes` should list at least one node)
- Network access from the cluster to the RTSP camera source

## Mock Stream for Testing

docker run --rm --name rfdetr-rtsp  -p 8554:8554/tcp \
  -e MTX_RTSPTRANSPORTS=tcp \
  -v "$PWD/mock-stream/mediamtx.yml:/mediamtx.yml:ro" \
  -v "$PWD/mock-stream:/videos:ro" \
  bluenviron/mediamtx:1.19.3-ffmpeg

## 1. Build the image

```bash
docker build -t x/rfdetr-inference:v1 .
```

Use `--platform linux/amd64` unless you know the cluster nodes run a
different architecture (e.g. ARM).

## 2. Push the image to a registry

```bash
docker push x/rfdetr-inference:v1
```

The Kubernetes cluster must be able to pull from this registry. If using a
private registry, make sure the cluster has an `imagePullSecret` configured
— ask your cluster admin if unsure.

## 3. Configure the RTSP source

Open `deployment.yaml` and set the camera URL:

```yaml
env:
- name: RTSP_URL
  value: "rtsp://<camera-host>:8554/<path>"
```

Also update the image reference to match what you pushed:

```yaml
image: <your-registry>/rfdetr-inference:v1
```

## 4. Deploy

```bash
colima start --cpu 4 --memory 6
minikube start --driver=docker --cpus=4 --memory=5911
minikube addons enable metrics-server
minikube image load x/rfdetr-inference:v1
kubectl apply -f deployment.yaml
kubectl apply -f prometheus.yaml
kubectl apply -f grafana.yaml
kubectl port-forward -n rfdetr svc/prometheus 9090:9090               
kubectl port-forward -n rfdetr svc/grafana 3000:3000
kubectl top pods -n rfdetr
# To debug:
# kubectl get pods -A     
# kubectl logs -n rfdetr -l app=rfdetr-inference -f 
# kubectl delete namespace rfdetr
```

This creates:
- Namespace: `rfdetr`
- Deployment: `rfdetr-inference` (1 replica, auto-restarts on failure)
- Service: `rfdetr-inference` (internal ClusterIP on port 8080)

## 5. Verify it's running

```bash
kubectl get pods -n rfdetr -w
```

Wait for `STATUS` to show `Running`. Press `Ctrl+C` to stop watching.

If it doesn't come up, check:

```bash
kubectl describe pod -n rfdetr <pod-name>   # events, image pull errors, probe failures
kubectl logs -n rfdetr <pod-name>           # application logs
```

## 6. Test the endpoints

Forward the service to your local machine:

```bash
kubectl port-forward -n rfdetr svc/rfdetr-inference 8080:8080
```

In another terminal:

```bash
curl http://localhost:8080/healthz   # should return "ok"
curl http://localhost:8080/metrics   # should return Prometheus metrics text
```

## 7. Monitoring / Prometheus

The Service is annotated for auto-discovery:

```yaml
prometheus.io/scrape: "true"
prometheus.io/port: "8080"
prometheus.io/path: "/metrics"
```

If the cluster already runs Prometheus with annotation-based discovery
enabled, no further action is needed. If it uses the Prometheus Operator
instead, a `ServiceMonitor` resource should be created pointing at this
Service — ask your monitoring team which pattern is in use.

Key metrics exposed:
- `frames_processed_total` — total frames processed
- `inference_latency_seconds` — inference time histogram
- `detections_total` — total objects detected across all frames

## 8. Confirm automatic recovery

To confirm the service restarts automatically if it fails or is killed:

```bash
kubectl delete pod -n rfdetr -l app=rfdetr-inference
kubectl get pods -n rfdetr -w
```

A new pod should appear and reach `Running` within a few seconds — no
manual intervention required.

## 9. Updating to a new version

```bash
docker build --platform linux/amd64 -t <your-registry>/rfdetr-inference:v2 .
docker push <your-registry>/rfdetr-inference:v2
```

Update the `image:` tag in `deployment.yaml` to `v2`, then:

```bash
kubectl apply -f deployment.yaml
```

Kubernetes performs a rolling update automatically.

## 10. Rolling back

```bash
kubectl rollout undo deployment/rfdetr-inference -n rfdetr
```

## Notes on resource sizing

The current `deployment.yaml` requests 1 CPU / 1Gi memory (limit 2 CPU / 2Gi).
This service runs RF-DETR inference on CPU, which is significantly slower
than GPU inference — expect inference times in the hundreds of milliseconds
to low seconds per frame depending on the node's CPU. Adjust the `resources`
block based on observed load and the cluster's available capacity.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Pod stuck in `ImagePullBackOff` | Image not pushed, wrong tag, or missing registry credentials on the cluster |
| Pod stuck in `CrashLoopBackOff` | Check `kubectl logs`; often an RTSP connection failure or missing dependency |
| `/healthz` returns 503 "stale" | No new frames received in 15s — check RTSP source is reachable from the cluster |
| Liveness probe failing right after startup | Model load takes time; increase `initialDelaySeconds` in `deployment.yaml` if needed |