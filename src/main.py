#!/usr/bin/env python3
"""
Pipeline: PLY -> Merge -> Preprocess -> Poisson mesh -> OBJ for Unity
Compatible with Open3D 0.19.0
"""

import os
import glob
import argparse
import open3d as o3d
import numpy as np

# -----------------------
# Utility functions
# -----------------------
def preprocess_ply(path, voxel_size=0.01, nb_neighbors=30, std_ratio=2.0,
                   normal_radius=0.05, normal_max_nn=30):
    print(f"[I] Loading {path}")
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {path}")

    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
    try:
        pcd.orient_normals_consistent_tangent_plane(100)
    except Exception:
        pass
    return pcd


def center_and_scale(pcd, padding=0.02):
    aabb_min = pcd.get_min_bound()
    aabb_max = pcd.get_max_bound()
    center = pcd.get_center()
    extent = aabb_max - aabb_min
    max_extent = max(extent)
    scale = 1.0 / (max_extent * (1.0 + padding))
    pcd.translate(-center)
    pcd.scale(scale, center=(0, 0, 0))
    print(f"[I] Centered and scaled with scale={scale:.6f}")


def merge_pointclouds(pcd_list):
    merged = o3d.geometry.PointCloud()
    points, colors, normals = [], [], []
    for pcd in pcd_list:
        points.extend(pcd.points)
        colors.extend(pcd.colors if pcd.has_colors() else [[0.5, 0.5, 0.5]] * len(pcd.points))
        normals.extend(pcd.normals if pcd.has_normals() else [[0, 0, 1]] * len(pcd.points))
    merged.points = o3d.utility.Vector3dVector(points)
    merged.colors = o3d.utility.Vector3dVector(colors)
    merged.normals = o3d.utility.Vector3dVector(normals)
    print(f"[I] Merged cloud has {len(points)} points")
    return merged


# -----------------------
# Pipeline
# -----------------------
def run_pipeline(
        data_folder=r"C:\Users\tamil\Downloads\wikki_temp\3dgs\Gaussian_splatting\data",
        out_dir=r"C:\Users\tamil\Downloads\wikki_temp\3dgs\Gaussian_splatting\output",
        voxel_size=0.01):

    os.makedirs(out_dir, exist_ok=True)

    ply_files = glob.glob(os.path.join(data_folder, "*.ply"))
    if not ply_files:
        raise RuntimeError(f"No PLY files found in {data_folder}")
    print(f"[I] Found {len(ply_files)} PLY files")

    processed = [preprocess_ply(f, voxel_size=voxel_size) for f in ply_files]
    merged = merge_pointclouds(processed)
    center_and_scale(merged)

    merged_path = os.path.join(out_dir, "merged_preprocessed.ply")
    o3d.io.write_point_cloud(merged_path, merged)
    print(f"[I] Saved merged point cloud -> {merged_path}")

    # -----------------------
    # Poisson Surface Reconstruction
    # -----------------------
    print("[I] Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        merged, depth=9, width=0, scale=1.1, linear_fit=True
    )

    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.075)
    vertices_to_keep = densities > density_threshold
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    mesh.compute_vertex_normals()

    mesh_path = os.path.join(out_dir, "poisson_mesh.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"[I] Saved mesh -> {mesh_path}")
    print("[I] Pipeline complete. Import .obj into Unity via Assets → Import New Asset")
    return mesh_path


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLY folder -> Poisson mesh -> Unity")
    parser.add_argument("--data",
                        default=r"C:\Users\tamil\Downloads\wikki_temp\3dgs\Gaussian_splatting\data",
                        help="Folder containing .ply files")
    parser.add_argument("--out",
                        default=r"C:\Users\tamil\Downloads\wikki_temp\3dgs\Gaussian_splatting\output",
                        help="Output folder")
    parser.add_argument("--voxel", type=float, default=0.01, help="Voxel downsample size")
    args = parser.parse_args()

    run_pipeline(
        data_folder=args.data,
        out_dir=args.out,
        voxel_size=args.voxel
    )
