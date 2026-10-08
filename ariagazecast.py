import csv
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import open3d as o3d
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence, get_nearest_pose
from scipy.spatial import cKDTree

INV_DIST_STD, DIST_STD = 0.005, 0.01
RADIUS_M = 50.0
POISSON_DEPTH = 10
TRIM_QUANTILE = 0.05
MAX_POSE_GAP_US = 1000


def build_mesh(points, glasses_positions):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    normals = np.asarray(pcd.normals)
    to_glasses = glasses_positions[cKDTree(glasses_positions).query(points)[1]] - points
    normals[(normals * to_glasses).sum(axis=1) < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=POISSON_DEPTH)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, TRIM_QUANTILE))
    return mesh.crop(pcd.get_axis_aligned_bounding_box())


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python ariagazecast.py RECORDING.vrs   (MPS output must be in mps_RECORDING_vrs/ next to it)")
    vrs = Path(sys.argv[1])
    slam = vrs.with_name(f"mps_{vrs.stem}_vrs") / "slam"
    out = vrs.with_name(vrs.stem + "_gaze_points.csv")
    mesh_out = vrs.with_name(vrs.stem + "_gaze_mesh.glb")
    if not vrs.is_file():
        sys.exit(f"{vrs}: not found")
    if not slam.is_dir():
        sys.exit(f"{slam}: not found (put the MPS output for {vrs.name} next to it, as {slam.parent.name}/)")

    provider = data_provider.create_vrs_data_provider(str(vrs))
    stream = provider.get_stream_id_from_label("eyegaze")
    if stream is None or provider.get_num_data(stream) == 0:
        sys.exit(f"{vrs}: no on-device eye gaze stream (record with a profile that has eye gaze)")

    trajectory = mps.read_closed_loop_trajectory(str(slam / "closed_loop_trajectory.csv"))
    points = filter_points_from_confidence(
        mps.read_global_point_cloud(str(slam / "semidense_points.csv.gz")), INV_DIST_STD, DIST_STD
    )
    xyz = np.array([p.position_world for p in points])
    xyz = xyz[np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1) <= RADIUS_M]
    glasses = np.array([p.transform_world_device.translation()[0] for p in trajectory[::100]])
    mesh = build_mesh(xyz, glasses)
    print(f"mesh: {len(mesh.triangles):,} triangles from {len(xyz):,} semi-dense points")
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(mesh_out), mesh):
        sys.exit(f"{mesh_out}: could not write the mesh")
    print(f"wrote {mesh_out}")

    T_device_cpf = provider.get_device_calibration().get_transform_device_cpf()
    times, rays = [], []
    for i in range(provider.get_num_data(stream)):
        gaze = provider.get_eye_gaze_data_by_index(stream, i)
        t_us = gaze.tracking_timestamp // timedelta(microseconds=1)
        times.append(t_us)
        pose = get_nearest_pose(trajectory, t_us * 1000)
        if not gaze.combined_gaze_valid or pose is None or abs(pose.tracking_timestamp // timedelta(microseconds=1) - t_us) > MAX_POSE_GAP_US:
            rays.append([0, 0, 0, 0, 0, 0])
            continue
        T_world_cpf = (pose.transform_world_device @ T_device_cpf).to_matrix()
        R, t = T_world_cpf[:3, :3], T_world_cpf[:3, 3]
        origin = R @ np.asarray(gaze.combined_gaze_origin_in_cpf).ravel() + t
        direction = R @ mps.get_eyegaze_point_at_depth(gaze.yaw, gaze.pitch, 1.0).ravel()
        rays.append([*origin, *direction])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.array(rays, dtype=np.float32)
    distance = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    hit = np.isfinite(distance) & (np.abs(rays[:, 3:]).sum(axis=1) > 0)
    point = rays[:, :3] + np.where(hit, distance, 0)[:, None] * rays[:, 3:]

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tracking_timestamp_us", "hit", "x_m", "y_m", "z_m", "distance_m"])
        for t, h, p, d in zip(times, hit, point, distance):
            writer.writerow([t, 1, *np.round(p, 4), round(float(d), 4)] if h else [t, 0, "", "", "", ""])
    print(f"gaze samples: {len(times):,}, hits: {int(hit.sum()):,}"
          + (f", median distance: {np.median(distance[hit]):.2f} m" if hit.any() else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
