import csv
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import get_nearest_pose


def read_mesh(mesh_path):
    if not mesh_path.is_file():
        sys.exit(f"{mesh_path}: not found (run ariagazecast.py first; it writes the mesh next to the CSV)")
    data = mesh_path.read_bytes()
    if data[:4] != b"glTF":
        sys.exit(f"{mesh_path}: not a GLB file")
    try:
        gltf = json.loads(data[20:20 + int.from_bytes(data[12:16], "little")])
        triangles = sum(
            gltf["accessors"][p["indices"]]["count"] // 3
            for m in gltf["meshes"] for p in m["primitives"] if "indices" in p
        )
    except (ValueError, KeyError, IndexError, TypeError) as e:
        sys.exit(f"{mesh_path}: could not read the mesh ({e!r})")
    if triangles == 0:
        sys.exit(f"{mesh_path}: the mesh has no triangles")
    return data, triangles


def log_gaze_points(csv_path, slam, mesh, triangles):
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    hits = [r for r in rows if r["hit"] == "1"]
    if not hits:
        sys.exit(f"{csv_path}: no gaze hits to show")
    trajectory = mps.read_closed_loop_trajectory(str(slam / "closed_loop_trajectory.csv"))
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
    rr.log("world/mesh", rr.Asset3D(contents=mesh, media_type="model/gltf-binary", albedo_factor=[170, 170, 170]), static=True)
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
    print(f"{len(rows):,} gaze samples, {len(hits):,} hits, mesh with {triangles:,} triangles")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python viewpoints.py RECORDING_gaze_points.csv   (MPS output must be in mps_RECORDING_vrs/ next to it)")
    csv_path = Path(sys.argv[1])
    recording = csv_path.name.removesuffix("_gaze_points.csv")
    slam = csv_path.with_name(f"mps_{recording}_vrs") / "slam"
    if not slam.is_dir():
        sys.exit(f"{slam}: not found (put the MPS output for {recording}.vrs next to the CSV, as {slam.parent.name}/)")
    mesh, triangles = read_mesh(csv_path.with_name(f"{recording}_gaze_mesh.glb"))
    rr.init("viewpoints", spawn=True)
    log_gaze_points(csv_path, slam, mesh, triangles)


if __name__ == "__main__":
    main()
