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
    boundary_key_arr = unique_keys[counts == 1]

    edges_unique = mesh.edges_unique
    edges_unique_sorted = np.sort(edges_unique, axis=1)
    unique_edge_keys = edges_unique_sorted[:, 0].astype(np.int64) * n_verts + edges_unique_sorted[:, 1]

    boundary_mask = np.in1d(unique_edge_keys, boundary_key_arr)
    boundary_edges = edges_unique[boundary_mask] if boundary_mask.any() else np.empty((0, 2), dtype=np.int64)

    parts = [sharp_edges, boundary_edges]
    parts = [p for p in parts if p.shape[0] > 0]

    if len(parts) == 0:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.concatenate(parts, axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    return edges


def _allocate_points_per_edge(
    edge_lengths: np.ndarray,
    total_length: float,
    num_points: int,
) -> np.ndarray:
    n_edges = len(edge_lengths)

    if num_points <= n_edges:
        idx = np.argpartition(edge_lengths, -num_points)[-num_points:]
        points_per_edge = np.zeros(n_edges, dtype=np.intp)
        points_per_edge[idx] = 1
        return points_per_edge

    points_per_edge = np.ones(n_edges, dtype=np.intp)
    remaining = num_points - n_edges
    extra = edge_lengths * (remaining / total_length)
    extra_int = extra.astype(np.intp)
    points_per_edge += extra_int

    deficit = num_points - points_per_edge.sum()
    if deficit > 0:
        fractional = extra - extra_int
        top_idx = np.argpartition(fractional, -deficit)[-deficit:]
        points_per_edge[top_idx] += 1

    return points_per_edge


def _interpolate_edges_vectorized(
    start: np.ndarray,
    end: np.ndarray,
    points_per_edge: np.ndarray,
) -> np.ndarray:
    '''Vectorized edge interpolation without Python loops.'''
    valid_mask = points_per_edge > 0
    if not valid_mask.any():
        return np.empty((0, 3), dtype=np.float64)

    ppe = points_per_edge[valid_mask]
    s = start[valid_mask]
    e = end[valid_mask]
    n_valid = len(ppe)
    total_pts = ppe.sum()

    # edge_idx[k] = which edge the k-th output point belongs to
    edge_idx = np.repeat(np.arange(n_valid, dtype=np.intp), ppe)

    # local_idx[k] = index within its edge (0-based)
    offsets = np.empty(n_valid + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(ppe, out=offsets[1:])
    local_idx = np.arange(total_pts, dtype=np.float64) - offsets[edge_idx].astype(np.float64)

    # t = (local_idx + 1) / (n + 1), where n = ppe for each edge
    denominators = (ppe + 1).astype(np.float64)
    t = (local_idx + 1.0) / denominators[edge_idx]
    t = t[:, np.newaxis]

    points = (1.0 - t) * s[edge_idx] + t * e[edge_idx]
    return points


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

    points_per_edge = _allocate_points_per_edge(edge_lengths, total_length, num_points)

    return _interpolate_edges_vectorized(start, end, points_per_edge)
