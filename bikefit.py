#!/usr/bin/env python3
"""Bike fit analysis from a side-view video using MediaPipe Pose."""

import argparse
import csv
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_heavy.task")

# MediaPipe landmark indices (left, right)
IDX = {
    "shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
    "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28), "toe": (31, 32),
}

# Target ranges in degrees (interior angles)
TARGET_BDC = (145.0, 155.0)
TARGET_TDC = (70.0, 110.0)
TARGET_ELBOW = (150.0, 170.0)
TOLERANCE = 3.0

FIELDS = ["frame", "knee", "hip", "torso", "elbow", "toe", "thigh", "shank"]


def angle(a, b, c):
    """Interior angle at b, in degrees."""
    ba, bc = np.asarray(a, float) - b, np.asarray(c, float) - b
    n = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n == 0:
        return float("nan")
    return math.degrees(math.acos(np.clip(np.dot(ba, bc) / n, -1, 1)))


def torso_angle(hip, shoulder):
    """Angle of hip-shoulder line to the horizontal."""
    return math.degrees(math.atan2(hip[1] - shoulder[1], abs(shoulder[0] - hip[0])))


def smooth(x, window):
    x = np.asarray(x, float)
    if window < 2:
        return x.copy()
    valid = ~np.isnan(x)
    k = np.ones(window)
    total = np.convolve(np.where(valid, x, 0.0), k, mode="same")
    count = np.convolve(valid.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        y = total / count
    y[count < window / 2] = np.nan
    return y


def extrema(sig, min_gap, maxima=True):
    """Local extrema at least min_gap frames apart, strongest first."""
    s = np.asarray(sig, float) if maxima else -np.asarray(sig, float)
    cand = [i for i in range(1, len(s) - 1)
            if not np.isnan(s[i]) and s[i] >= np.nanmax(s[i - 1:i + 2])]
    cand.sort(key=lambda i: s[i], reverse=True)
    chosen = []
    for i in cand:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def pedal_period(sig, fps):
    """Frames per crank revolution, from autocorrelation."""
    s = np.asarray(sig, float)
    s = s[~np.isnan(s)]
    default = int(fps * 0.7)
    if len(s) < fps:
        return default
    s = s - s.mean()
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    lo, hi = int(fps * 0.4), min(int(fps * 1.5), len(ac) - 1)
    return lo + int(np.argmax(ac[lo:hi])) if hi > lo else default


def load_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading pose model (~30 MB, once) ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        except Exception as e:
            sys.exit(f"Download failed: {e}\nGet it manually and place it next to this script:\n{MODEL_URL}")
    return MODEL_FILE


def detect_poses(video):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=load_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    poses = []  # per frame: (33, 3) array of x, y, visibility, or None
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(img, int(len(poses) * 1000 / fps))
            poses.append(np.array([[q.x * w, q.y * h, q.visibility] for q in res.pose_landmarks[0]])
                         if res.pose_landmarks else None)
            if total and len(poses) % 30 == 0:
                print(f"\r  frame {len(poses)}/{total}", end="", flush=True)
    cap.release()
    print(f"\r  frame {len(poses)}/{len(poses)}")
    return poses, fps


def pick_side(poses):
    """Side facing the camera = higher landmark visibility."""
    score = [0.0, 0.0]
    for p in poses:
        if p is not None:
            for s in (0, 1):
                score[s] += np.mean([p[IDX[k][s], 2] for k in ("hip", "knee", "ankle")])
    return 0 if score[0] >= score[1] else 1


def measure(poses, s, min_vis=0.5):
    rows = []
    for i, p in enumerate(poses):
        r = dict.fromkeys(FIELDS, np.nan)
        r["frame"] = i
        if p is not None and min(p[IDX[k][s], 2] for k in ("hip", "knee", "ankle")) >= min_vis:
            pt = {k: p[IDX[k][s], :2] for k in IDX}
            r.update(
                knee=angle(pt["hip"], pt["knee"], pt["ankle"]),
                hip=angle(pt["shoulder"], pt["hip"], pt["knee"]),
                torso=torso_angle(pt["hip"], pt["shoulder"]),
                elbow=angle(pt["shoulder"], pt["elbow"], pt["wrist"]),
                toe=math.degrees(math.atan2(pt["toe"][1] - pt["ankle"][1],
                                            abs(pt["toe"][0] - pt["ankle"][0]))),
                thigh=float(np.linalg.norm(pt["hip"] - pt["knee"])),
                shank=float(np.linalg.norm(pt["knee"] - pt["ankle"])),
            )
        rows.append(r)
    return rows


def drop_outliers(rows, idx):
    """Drop revolutions whose knee angle is far from the median (3x MAD, at least 4 deg)."""
    v = np.array([rows[i]["knee"] for i in idx])
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return [i for i, g in zip(idx, ok) if g]
    med = np.median(v[ok])
    limit = max(4.0, 3 * 1.4826 * np.median(np.abs(v[ok] - med)))
    return [i for i, x in zip(idx, v) if not np.isnan(x) and abs(x - med) <= limit]


def analyze(rows, fps, height_cm=None):
    knee = np.array([r["knee"] for r in rows])
    valid = int(np.sum(~np.isnan(knee)))
    if valid < fps:
        sys.exit("Too few frames with a clearly detected leg. Use a true side view, "
                 "full body in frame, good light.")

    period = pedal_period(knee, fps)
    sig = smooth(knee, max(1, int(fps / 20)))
    gap = max(2, int(period * 0.6))
    bdc = drop_outliers(rows, extrema(sig, gap, maxima=True))
    tdc = drop_outliers(rows, extrema(sig, gap, maxima=False))

    def med(key, idx):
        v = [rows[i][key] for i in idx if not np.isnan(rows[i][key])]
        return float(np.median(v)) if v else float("nan")

    def spread(idx):
        v = [rows[i]["knee"] for i in idx]
        return float(np.std(v)) if v else float("nan")

    res = dict(
        frames=len(rows), valid=valid, fps=fps, cadence=60.0 * fps / period,
        bdc=bdc, tdc=tdc,
        bdc_knee=med("knee", bdc), bdc_std=spread(bdc),
        tdc_knee=med("knee", tdc), tdc_std=spread(tdc),
        tdc_hip=med("hip", tdc), toe_bdc=med("toe", bdc),
        torso=float(np.nanmedian([r["torso"] for r in rows])),
        elbow=float(np.nanmedian([r["elbow"] for r in rows])),
    )

    def closest(idx, target):
        return min(idx, key=lambda i: abs(rows[i]["knee"] - target)) if idx else None

    res["bdc_frame"] = closest(bdc, res["bdc_knee"])
    res["tdc_frame"] = closest(tdc, res["tdc_knee"])

    # Saddle height change via law of cosines: hip-ankle distance at measured vs. target angle
    if bdc:
        t, sh = med("thigh", bdc), med("shank", bdc)

        def dist(deg):
            return math.sqrt(t**2 + sh**2 - 2 * t * sh * math.cos(math.radians(deg)))

        delta = dist(res["bdc_knee"]) - dist(sum(TARGET_BDC) / 2)  # > 0: lower saddle
        res["delta_pct"] = 100 * delta / (t + sh)
        if height_cm:
            res["delta_mm"] = delta * (0.491 * height_cm * 10) / (t + sh)  # thigh + shank ~ 0.491 * height
    return res


def rate(value, target):
    if math.isnan(value):
        return "n/a"
    lo, hi = target
    if lo <= value <= hi:
        return "ok"
    if lo - TOLERANCE <= value <= hi + TOLERANCE:
        return "borderline"
    return "too low" if value < lo else "too high"


def report(res, side):
    def line(name, value, target=None, std=None):
        rng = f"{target[0]:.0f}-{target[1]:.0f}" if target else ""
        sd = f"±{std:.1f}" if std is not None else ""
        verdict = rate(value, target) if target else ""
        return f"{name:<16}{value:>6.1f}°  {sd:<7}{rng:<10}{verdict}"

    out = [
        "BIKE FIT ANALYSIS",
        f"Side: {side} | {res['valid']}/{res['frames']} frames at {res['fps']:.0f} fps | "
        f"cadence ~{res['cadence']:.0f} rpm | revolutions: {len(res['bdc'])}",
        "",
        f"{'':<16}{'median':>7}  {'spread':<7}{'target':<10}rating",
        line("Knee bottom", res["bdc_knee"], TARGET_BDC, res["bdc_std"]),
        line("Knee top", res["tdc_knee"], TARGET_TDC, res["tdc_std"]),
        line("Elbow", res["elbow"], TARGET_ELBOW),
        line("Hip (top)", res["tdc_hip"]),
        line("Torso", res["torso"]),
        line("Toe (bottom)", res["toe_bdc"]),
        "",
    ]

    verdict = rate(res["bdc_knee"], TARGET_BDC)
    if verdict == "n/a":
        out.append("Saddle: no bottom dead center found.")
    elif verdict in ("ok", "borderline"):
        out.append("Saddle: height is fine, leave it.")
    else:
        direction = "lower" if res["delta_pct"] > 0 else "raise"
        size = (f"~{abs(res['delta_mm']):.0f} mm" if "delta_mm" in res
                else f"~{abs(res['delta_pct']):.1f}% of leg length (pass --height for mm)")
        out.append(f"Saddle: {direction} by {size}. Change in 5 mm steps, then film again.")
    if res["bdc_std"] > 4:
        out.append(f"Note: knee angle varies a lot ({res['bdc_std']:.1f}°). Check light and camera angle.")
    out.append("")
    out.append("2D pose estimation is typically off by 3-9°. This does not replace a professional fit.")
    return "\n".join(out)


def draw(frame, pose, s, label):
    if pose is None:
        return frame
    for chain in (("shoulder", "hip", "knee", "ankle", "toe"), ("shoulder", "elbow", "wrist")):
        pts = [tuple(int(v) for v in pose[IDX[k][s], :2]) for k in chain]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, (0, 220, 255), 4, cv2.LINE_AA)
        for q in pts:
            cv2.circle(frame, q, 7, (0, 0, 255), -1, cv2.LINE_AA)
    h, k, a = (pose[IDX[n][s], :2] for n in ("hip", "knee", "ankle"))
    text = f"{angle(h, k, a):.0f}"
    pos = (int(k[0]) + 15, int(k[1]) + 15)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3, cv2.LINE_AA)
    return frame


def write_outputs(video, poses, s, rows, res, outdir, with_video, preview):
    cap = cv2.VideoCapture(video)
    writer = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pose = poses[i] if i < len(poses) else None
        if i == res["bdc_frame"]:
            cv2.imwrite(os.path.join(outdir, "bottom.png"),
                        draw(frame.copy(), pose, s, f"Bottom  {res['bdc_knee']:.0f} deg"))
        if i == res["tdc_frame"]:
            cv2.imwrite(os.path.join(outdir, "top.png"),
                        draw(frame.copy(), pose, s, f"Top  {res['tdc_knee']:.0f} deg"))
        if with_video or preview:
            label = "BOTTOM" if i in res["bdc"] else "TOP" if i in res["tdc"] else None
            annotated = draw(frame, pose, s, label)
            if with_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(outdir, "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), res["fps"], (w, h))
                writer.write(annotated)
            if preview:
                cv2.imshow("bikefit (q to stop preview)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    preview = False
                    cv2.destroyAllWindows()
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    if writer:
        writer.release()

    with open(os.path.join(outdir, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description="Bike fit analysis from a side-view video")
    ap.add_argument("video")
    ap.add_argument("--side", choices=["left", "right"], help="body side facing the camera (default: auto)")
    ap.add_argument("--height", type=float, help="body height in cm, gives saddle change in mm")
    ap.add_argument("--preview", type=lambda v: v.lower() not in ("false", "0", "no"), default=True,
                    metavar="true|false", help="show a live preview while rendering (default: true)")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video")
    a = ap.parse_args()

    outdir = os.path.splitext(a.video)[0] + "_bikefit"
    os.makedirs(outdir, exist_ok=True)

    print("Detecting pose ...")
    poses, fps = detect_poses(a.video)
    s = {"left": 0, "right": 1}.get(a.side) if a.side else pick_side(poses)

    rows = measure(poses, s)
    res = analyze(rows, fps, a.height)
    text = report(res, "left" if s == 0 else "right")
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(text + "\n")

    print("Writing outputs ...")
    write_outputs(a.video, poses, s, rows, res, outdir, not a.no_video, a.preview)

    print(f"\n{text}\n\nResults in: {outdir}/")


if __name__ == "__main__":
    main()
