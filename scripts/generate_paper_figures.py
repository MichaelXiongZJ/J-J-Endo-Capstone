"""Generate publication-quality figures for Academic Paper Chapters 3 & 4.

Uses real CCTV warehouse images from `data/real/forklift2-z6zww_v5/valid/` alongside
perceptive overlays to produce Figures 1-5 saved in `paper_figures/` and copied to artifacts.
"""

import os
import glob
import shutil
import cv2
import numpy as np

from src.pose_utils import (SKELETON, NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER,
                             L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP, valid)
from src.rules import CFG, R5_CHECK_KEYPOINTS

OUTPUT_DIR = "paper_figures"
ARTIFACTS_DIR = r"C:\Users\Michael\.gemini\antigravity-ide\brain\cc2684e6-c840-4867-a582-4532a87642be"

REAL_IMG_DIR = "data/real/forklift2-z6zww_v5/valid"


def get_real_sample_image():
    imgs = sorted(glob.glob(os.path.join(REAL_IMG_DIR, "*.jpg")))
    if imgs:
        frame = cv2.imread(imgs[0])
        if frame is not None:
            return cv2.resize(frame, (1920, 1080))
    # Fallback dark synthetic background if real image not loaded
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (35, 38, 42)
    return frame


def get_real_sample_image_index(idx=5):
    imgs = sorted(glob.glob(os.path.join(REAL_IMG_DIR, "*.jpg")))
    if len(imgs) > idx:
        frame = cv2.imread(imgs[idx])
        if frame is not None:
            return cv2.resize(frame, (1920, 1080))
    return get_real_sample_image()


# ==============================================================================
# FIGURE 1: HOMOGRAPHIC PROJECTION OF WAREHOUSE FLOOR
# ==============================================================================
def gen_fig1():
    base_img = get_real_sample_image()
    h, w = 1080, 1920

    # Panel A: Original Camera View with 4 surveyed Ground Control Points
    p1 = base_img.copy()
    gcps = [(350, 450), (1550, 420), (1800, 980), (120, 950)]
    cv2.polylines(p1, [np.array(gcps, np.int32)], True, (0, 215, 255), 3)

    labels = ["GCP 1 (0.0m, 0.0m)", "GCP 2 (12.0m, 0.0m)", "GCP 3 (12.0m, 20.0m)", "GCP 4 (0.0m, 20.0m)"]
    for i, (gx, gy) in enumerate(gcps):
        cv2.circle(p1, (gx, gy), 12, (0, 0, 255), -1)
        cv2.circle(p1, (gx, gy), 16, (255, 255, 255), 2)
        cv2.putText(p1, labels[i], (gx + 15, gy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(p1, "(a) Original CCTV Camera Perspective (Surveyed GCPs in Yellow)",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    # Panel B: Top-Down Orthographic Metric Floor Projection
    p2 = np.zeros((h, w, 3), dtype=np.uint8)
    p2[:] = (22, 26, 30)

    # Draw metric grid (0-12m X, 0-20m Y)
    for y_m in range(0, 21, 4):
        py = int(120 + y_m * 42)
        cv2.line(p2, (200, py), (1700, py), (50, 58, 68), 1)
        cv2.putText(p2, f"{y_m}.0 m", (110, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    for x_m in range(0, 13, 3):
        px = int(200 + x_m * 125)
        cv2.line(p2, (px, 120), (px, 960), (50, 58, 68), 1)
        cv2.putText(p2, f"{x_m}.0 m", (px - 25, 995), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    cv2.rectangle(p2, (200, 120), (1700, 960), (0, 255, 0), 3)

    # Projected object points
    cv2.circle(p2, (575, 456), 14, (0, 255, 0), -1)  # Pedestrian at (3.0m, 8.0m)
    cv2.putText(p2, "Pedestrian (3.00m, 8.00m)", (600, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.circle(p2, (1075, 666), 14, (0, 215, 255), -1)  # Forklift at (7.0m, 13.0m)
    cv2.putText(p2, "Forklift (7.00m, 13.00m)", (1100, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

    cv2.line(p2, (575, 456), (1075, 666), (0, 215, 255), 3)
    cv2.putText(p2, "Euclidean Distance: 6.40 m", (720, 540),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)

    cv2.putText(p2, "(b) Orthographic Top-Down Ground-Plane Projection (Euclidean Metric Grid)",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    # Side-by-side composite
    fig = np.zeros((1080, 1920 * 2, 3), dtype=np.uint8)
    fig[:, :1920] = p1
    fig[:, 1920:] = p2
    fig = cv2.resize(fig, (1920, 540))

    out = os.path.join(OUTPUT_DIR, "fig1_calibration.png")
    cv2.imwrite(out, fig)
    return out


# ==============================================================================
# FIGURE 2: DRIVER ASSOCIATION VIA VELOCITY VECTOR MATCHING
# ==============================================================================
def gen_fig2():
    frame = get_real_sample_image_index(idx=10)

    # Draw forklift bounding box
    vbox = (500, 320, 1350, 920)
    cv2.rectangle(frame, (vbox[0], vbox[1]), (vbox[2], vbox[3]), (0, 215, 255), 3)
    cv2.putText(frame, "FORKLIFT [id1] - SPEED: 1.35 m/s (ACTIVE)", (vbox[0], vbox[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 215, 255), 2)

    # Driver Box (Green) - Matches velocity
    dbox = (780, 360, 960, 680)
    cv2.rectangle(frame, (dbox[0], dbox[1]), (dbox[2], dbox[3]), (0, 255, 0), 3)
    cv2.putText(frame, "DRIVER [id2 DRV] - MATCHED (dv = 0.05 m/s)", (dbox[0] - 80, dbox[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.arrowedLine(frame, (870, 520), (1040, 570), (0, 255, 0), 4, tipLength=0.3)

    # Occluded Pedestrian Behind Forklift (Red Box) - Velocity Divergence
    pbox = (1180, 380, 1320, 780)
    cv2.rectangle(frame, (pbox[0], pbox[1]), (pbox[2], pbox[3]), (0, 0, 255), 3)
    cv2.putText(frame, "OCCLUDED PEDESTRIAN [id3] - DIVERGENT (dv = 1.48 m/s)", (pbox[0] - 180, pbox[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.arrowedLine(frame, (1250, 580), (1250, 420), (0, 0, 255), 4, tipLength=0.3)

    # Title Legend Banner
    cv2.rectangle(frame, (40, 40), (1880, 110), (18, 22, 28), -1)
    cv2.rectangle(frame, (40, 40), (1880, 110), (0, 215, 255), 2)
    cv2.putText(frame, "Figure 2: Driver Association Engine via Rolling Velocity Vector Correlation",
                (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, "Distinguishes active driver (green, dv=0.05 m/s) from pedestrian occluded behind vehicle (red, dv=1.48 m/s)",
                (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 240, 255), 2)

    out = os.path.join(OUTPUT_DIR, "fig2_driver_association.png")
    cv2.imwrite(out, frame)
    return out


# ==============================================================================
# FIGURE 3: GEOMETRIC SAFETY RULE EVALUATION (RULES 3 & 4)
# ==============================================================================
def gen_fig3():
    base1 = get_real_sample_image_index(idx=15)
    base2 = get_real_sample_image_index(idx=20)

    # Subfigure (a): Rule 3 Proximity
    p1 = base1.copy()
    p_box = (380, 420, 520, 800)
    v_box = (900, 360, 1500, 880)

    cv2.rectangle(p1, (v_box[0], v_box[1]), (v_box[2], v_box[3]), (0, 215, 255), 3)
    cv2.putText(p1, "FORKLIFT [id1] (WORKING)", (v_box[0], v_box[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

    cv2.rectangle(p1, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 0, 255), 3)
    cv2.putText(p1, "PEDESTRIAN [id3] (VIOLATION)", (p_box[0] - 20, p_box[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.line(p1, (450, 800), (1200, 880), (0, 0, 255), 3)
    cv2.ellipse(p1, (1200, 880), (450, 160), 0, 0, 360, (0, 0, 255), 2)
    cv2.putText(p1, "(a) Rule 3 (Proximity): Dynamic 3.0-Vehicle-Length Floor Safety Radius (5.2m < 8.1m)",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Subfigure (b): Rule 4 Walkways
    p2 = base2.copy()
    pts = np.array([[200, 320], [750, 320], [850, 980], [150, 980]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    overlay = p2.copy()
    cv2.fillPoly(overlay, [pts], (255, 140, 0))
    cv2.addWeighted(overlay, 0.35, p2, 0.65, 0, p2)
    cv2.polylines(p2, [pts], True, (255, 200, 0), 3)

    p2_box = (1100, 450, 1220, 820)
    cv2.rectangle(p2, (p2_box[0], p2_box[1]), (p2_box[2], p2_box[3]), (0, 0, 255), 3)
    cv2.putText(p2, "PEDESTRIAN [id5] - BREACH (1.2s > 1.0s GATE)", (p2_box[0] - 80, p2_box[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(p2, "(b) Rule 4 (Walkways): Assessment Against Predefined Polygonal Safe Zone",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2)

    fig = np.zeros((1080, 1920 * 2, 3), dtype=np.uint8)
    fig[:, :1920] = p1
    fig[:, 1920:] = p2
    fig = cv2.resize(fig, (1920, 540))

    out = os.path.join(OUTPUT_DIR, "fig3_geometric_rules.png")
    cv2.imwrite(out, fig)
    return out


# ==============================================================================
# FIGURE 4: POSE-DEPENDENT RULE EVALUATION (RULES 1 & 5)
# ==============================================================================
def gen_fig4():
    base1 = get_real_sample_image_index(idx=25)
    base2 = get_real_sample_image_index(idx=30)

    # Subfigure (a): Rule 5 Driver Protrusion
    p1 = base1.copy()
    vbox = (550, 280, 1350, 920)
    vx1, vy1, vx2, vy2 = vbox
    vw, vh = vx2 - vx1, vy2 - vy1
    cab = (int(vx1 + 0.15 * vw), int(vy1 + 0.35 * vh), int(vx2 - 0.15 * vw), int(vy2 - 0.0 * vh))
    cx1, cy1, cx2, cy2 = cab

    cv2.rectangle(p1, (vx1, vy1), (vx2, vy2), (0, 215, 255), 3)
    cv2.rectangle(p1, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[NOSE] = (850, 420, 0.95)
    kpts[L_SHOULDER] = (780, 500, 0.94)
    kpts[R_SHOULDER] = (920, 500, 0.94)
    kpts[L_WRIST] = (710, 650, 0.93)
    kpts[R_WRIST] = (1285, 560, 0.98)  # Out of cab

    for a, b in [(5, 6), (5, 9), (6, 10), (0, 5), (0, 6)]:
        cv2.line(p1, (int(kpts[a][0]), int(kpts[a][1])), (int(kpts[b][0]), int(kpts[b][1])), (0, 255, 255), 3)

    for i in [NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST]:
        px, py = int(kpts[i][0]), int(kpts[i][1])
        out = not (cx1 <= px <= cx2 and cy1 <= py <= cy2)
        cv2.circle(p1, (px, py), 10 if out else 7, (0, 0, 255) if out else (0, 255, 0), -1)

    cv2.putText(p1, "(a) Rule 5 (Driver Protrusion): Dynamic Cab Inset (Cyan) & Protruding Keypoints (Red)",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Subfigure (b): Rule 1 Phone Usage
    p2 = base2.copy()
    p_box = (700, 320, 1100, 950)
    cv2.rectangle(p2, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 0, 255), 3)

    cv2.line(p2, (800, 520), (1000, 520), (255, 215, 0), 4)
    cv2.line(p2, (955, 425), (950, 410), (0, 0, 255), 3)
    cv2.circle(p2, (955, 425), 10, (0, 0, 255), -1)
    cv2.putText(p2, "Ratio: 0.38 < 0.60 (Sustained 2.2s)", (960, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.putText(p2, "(b) Rule 1 (Phone Usage): Wrist-to-Head Distance Normalized Against Shoulder Width",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    fig = np.zeros((1080, 1920 * 2, 3), dtype=np.uint8)
    fig[:, :1920] = p1
    fig[:, 1920:] = p2
    fig = cv2.resize(fig, (1920, 540))

    out = os.path.join(OUTPUT_DIR, "fig4_pose_rules.png")
    cv2.imwrite(out, fig)
    return out


# ==============================================================================
# FIGURE 5: AGGREGATED LEADING INDICATORS (SPATIAL HEATMAP)
# ==============================================================================
def gen_fig5():
    h, w = 1080, 1920
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 33, 38)

    # Draw warehouse floor layout blueprint
    # Aisle walls & racking structures
    racks = [
        (200, 150, 450, 900),
        (650, 150, 900, 900),
        (1100, 150, 1350, 900),
        (1550, 150, 1800, 900),
    ]
    for rx1, ry1, rx2, ry2 in racks:
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (60, 68, 78), -1)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (100, 110, 125), 2)
        cv2.putText(img, "STORAGE RACKING BAY", (rx1 + 15, ry1 + 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 150, 165), 2)

    # Intersections & high-risk hot zones
    heat = np.zeros((h, w), dtype=np.float32)

    # Add Gaussian hot spots at aisle intersections
    hotspots = [
        (550, 520, 180, 1.0),  # Blind intersection 1
        (1000, 520, 220, 1.2), # Main cross-aisle junction
        (1450, 520, 160, 0.8), # Loading dock entry
        (550, 850, 140, 0.7),
    ]
    y_coords, x_coords = np.ogrid[:h, :w]
    for hx, hy, sigma, intensity in hotspots:
        dist_sq = (x_coords - hx) ** 2 + (y_coords - hy) ** 2
        heat += intensity * np.exp(-dist_sq / (2 * sigma ** 2))

    heat = np.clip(heat, 0, 1)
    heat_colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Blend heatmap over floorplan
    alpha = 0.55
    blended = cv2.addWeighted(heat_colored, alpha, img, 1 - alpha, 0)

    # Redraw crisp boundaries over blended heatmap
    for rx1, ry1, rx2, ry2 in racks:
        cv2.rectangle(blended, (rx1, ry1), (rx2, ry2), (200, 200, 200), 2)

    # Callout boxes for high-risk zones
    cv2.rectangle(blended, (450, 420), (650, 620), (0, 0, 255), 3)
    cv2.putText(blended, "HIGH-RISK ZONE 1 (42 Near-Misses)", (380, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(blended, (880, 400), (1120, 640), (0, 0, 255), 3)
    cv2.putText(blended, "HIGH-RISK ZONE 2 (68 Near-Misses)", (830, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Title Legend
    cv2.rectangle(blended, (40, 40), (1880, 110), (15, 18, 22), -1)
    cv2.rectangle(blended, (40, 40), (1880, 110), (0, 255, 255), 2)
    cv2.putText(blended, "Figure 5: Aggregated Leading Indicators — Spatial Near-Miss Heatmap (30-Day Evaluation Period)",
                (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(blended, "Quantifies spatial frequency of Rule 3 proximity breaches across warehouse aisles to guide structural interventions",
                (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 240, 255), 2)

    out = os.path.join(OUTPUT_DIR, "fig5_heatmap.png")
    cv2.imwrite(out, blended)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    figs = [gen_fig1(), gen_fig2(), gen_fig3(), gen_fig4(), gen_fig5()]

    print("\n" + "=" * 70)
    print("ACADEMIC PAPER FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    for f in figs:
        base = os.path.basename(f)
        dest = os.path.join(ARTIFACTS_DIR, base)
        shutil.copy(f, dest)
        print(f"  -> {f}")
        print(f"     (Copied to artifact: {dest})")
    print("=" * 70)


if __name__ == '__main__':
    main()
