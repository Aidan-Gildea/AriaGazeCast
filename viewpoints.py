import csv
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence, get_nearest_pose

INV_DIST_STD, DIST_STD = 0.005, 0.01
RADIUS_M = 50.0


def log_gaze_points(csv_path, slam):
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    hits = [r for r in rows if r["hit"] == "1"]
    if not hits:
        sys.exit(f"{csv_path}: no gaze hits to show")
    trajectory = mps.read_closed_loop_trajectory(str(slam / "closed_loop_trajectory.csv"))
    points = filter_points_from_confidence(
        mps.read_global_point_cloud(str(slam / "semidense_points.csv.gz")), INV_DIST_STD, DIST_STD
    )
    xyz = np.array([p.position_world for p in points])
    xyz = xyz[np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1) <= RADIUS_M]
    path = np.array([p.transform_world_device.translation()[0] for p in trajectory[::100]])

    hit_t = np.array([int(r["tracking_timestamp_us"]) for r in hits])
    hit_xyz = np.array([[float(r["x_m"]), float(r["y_m"]), float(r["z_m"])] for r in hits])
    s = ((hit_t - hit_t.min()) / max(np.ptp(hit_t), 1))[:, None]
    hit_colors = ((1 - s) * [40, 120, 255] + s * [255, 90, 40]).astype(np.uint8)

    rr.send_blueprint(rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(origin="world", name="gaze points"),
            rrb.TimeSeriesView(origin="distance_m", name="gaze distance (m)"),
            row_shares=[4, 1],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    ))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/semidense_points", rr.Points3D(xyz, colors=[160, 160, 160], radii=rr.Radius.ui_points(1.0)), static=True)
    rr.log("world/glasses_path", rr.LineStrips3D([path], colors=[90, 200, 120], radii=rr.Radius.ui_points(1.5)), static=True)
    rr.log("world/gaze_points", rr.Points3D(hit_xyz, colors=hit_colors, radii=rr.Radius.ui_points(4.0)), static=True)

    for r in rows:
        t_us = int(r["tracking_timestamp_us"])
        rr.set_time("tracking_time", duration=t_us * 1e-6)
        if r["hit"] != "1":
            rr.log("world/gaze", rr.Clear(recursive=True))
            continue
        glasses = get_nearest_pose(trajectory, t_us * 1000).transform_world_device.translation()[0]
        point = [float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]
        rr.log("world/gaze/line", rr.LineStrips3D([[glasses, point]], colors=[255, 60, 160], radii=rr.Radius.ui_points(1.5)))
        rr.log("world/gaze/point", rr.Points3D([point], colors=[255, 60, 160], radii=rr.Radius.ui_points(8.0)))
        rr.log("distance_m", rr.Scalars(float(r["distance_m"])))
    print(f"{len(rows):,} gaze samples, {len(hits):,} hits, {len(xyz):,} semi-dense points")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python viewpoints.py RECORDING_gaze_points.csv   (MPS output must be in mps_RECORDING_vrs/ next to it)")
    csv_path = Path(sys.argv[1])
    recording = csv_path.name.removesuffix("_gaze_points.csv")
    slam = csv_path.with_name(f"mps_{recording}_vrs") / "slam"
    if not slam.is_dir():
        sys.exit(f"{slam}: not found (put the MPS output for {recording}.vrs next to the CSV, as {slam.parent.name}/)")
    rr.init("viewpoints", spawn=True)
    log_gaze_points(csv_path, slam)


if __name__ == "__main__":
    main()
