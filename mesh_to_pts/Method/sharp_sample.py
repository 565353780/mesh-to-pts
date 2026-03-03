import math
import trimesh
import numpy as np


def sampleSharpEdges(
    mesh: trimesh.Trimesh,
    angle_threshold: float,
) -> np.ndarray:
    '''
    return: [[v_start_idx_0, v_end_idx_0], ...] # Nx2 w/o repeat
    '''
    sharpness_threshold = math.radians(angle_threshold)

    face_adjacency = mesh.face_adjacency
    face_adjacency_edges = mesh.face_adjacency_edges
    face_normals = mesh.face_normals

    n0 = face_normals[face_adjacency[:, 0]]
    n1 = face_normals[face_adjacency[:, 1]]

    dot = np.einsum('ij,ij->i', n0, n1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.arccos(dot)

    sharp_mask = angles > sharpness_threshold
    sharp_edges = face_adjacency_edges[sharp_mask]

    edges_all = np.sort(mesh.edges, axis=1)
    n_verts = mesh.vertices.shape[0] + 1
    edge_keys = edges_all[:, 0].astype(np.int64) * n_verts + edges_all[:, 1]
    unique_keys, counts = np.unique(edge_keys, return_counts=True)
    boundary_keys = set(unique_keys[counts == 1].tolist())
    edges_unique = mesh.edges_unique
    edges_unique_sorted = np.sort(edges_unique, axis=1)
    unique_edge_keys = edges_unique_sorted[:, 0].astype(np.int64) * n_verts + edges_unique_sorted[:, 1]
    boundary_mask = np.isin(unique_edge_keys, list(boundary_keys))
    boundary_edges = edges_unique[boundary_mask] if boundary_mask.any() else np.empty((0, 2), dtype=np.int64)

    parts = [sharp_edges, boundary_edges]
    parts = [p for p in parts if p.shape[0] > 0]

    if len(parts) == 0:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.concatenate(parts, axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    return edges

def sampleSharpEdgePoints(
    mesh: trimesh.Trimesh,
    angle_threshold: float,
    num_points: int,
) -> np.ndarray:
    '''
    return: sharp_edge_points # Nx3
    '''
    sharp_edges = sampleSharpEdges(mesh, angle_threshold)

    if sharp_edges.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    vertices = mesh.vertices
    start = vertices[sharp_edges[:, 0]]
    end = vertices[sharp_edges[:, 1]]

    edge_lengths = np.linalg.norm(end - start, axis=1)
    total_length = edge_lengths.sum()

    if total_length < 1e-12:
        return vertices[sharp_edges[:, 0]]

    points_per_edge = np.maximum(
        np.round(edge_lengths / total_length * num_points).astype(int), 1
    )
    # 调整使总点数接近 num_points（舍入和至少 1 点/边可能导致总和偏差）
    current_total = points_per_edge.sum()
    if current_total != num_points and sharp_edges.shape[0] > 0:
        diff = num_points - current_total
        if diff > 0:
            idx = np.argsort(edge_lengths)[::-1]
            for i in range(min(diff, len(idx))):
                points_per_edge[idx[i]] += 1
        else:
            idx = np.argsort(edge_lengths)
            for i in range(min(-diff, len(idx))):
                j = idx[i]
                if points_per_edge[j] > 1:
                    points_per_edge[j] -= 1

    all_points = []
    for i in range(sharp_edges.shape[0]):
        n = points_per_edge[i]
        t = np.linspace(0.0, 1.0, n + 2)[1:-1].reshape(-1, 1)
        pts = (1.0 - t) * start[i] + t * end[i]
        all_points.append(pts)

    sharp_edge_points = np.concatenate(all_points, axis=0)

    unique_pts = np.unique(sharp_edge_points, axis=0)

    return unique_pts
