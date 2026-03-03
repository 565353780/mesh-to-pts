import trimesh
import numpy as np

from mesh_to_pts.Method.sharp_sample import sampleSharpEdgePoints

class PointSampler(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def sampleSurfPoints(
        mesh: trimesh.Trimesh,
        num_points: int,
    ) -> np.ndarray:
        """
        使用trimesh在mesh表面均匀采样num_points个点

        Args:
            mesh (trimesh.Trimesh): 输入三角网格
            num_points (int): 采样点数

        Returns:
            np.ndarray: [num_points, 3] 采样点坐标
        """
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points

    @staticmethod
    def sampleSharpEdgePoints(
        mesh: trimesh.Trimesh,
        angle_threshold: float,
        num_points: int,
    ) -> np.ndarray:
        return sampleSharpEdgePoints(
            mesh=mesh,
            angle_threshold=angle_threshold,
            num_points=num_points,
        )

    @staticmethod
    def dropPoints(
        source_points: np.ndarray,
        drop_ratio: float,
    ) -> np.ndarray:
        """
        随机从source_points中剔除drop_ratio比例的点，返回剩余点
        """
        num_points = source_points.shape[0]
        num_drop = int(np.floor(num_points * drop_ratio))
        if num_drop <= 0:
            return source_points
        if num_drop >= num_points:
            return np.empty((0, source_points.shape[1]), dtype=source_points.dtype)
        idx = np.arange(num_points)
        # 随机打乱下标
        np.random.shuffle(idx)
        keep_idx = idx[num_drop:]
        kept_points = source_points[keep_idx]
        return kept_points

    @staticmethod
    def _clippedBoxVolumeFraction(
        half_extents: np.ndarray,
        normal: np.ndarray,
        d: float,
    ) -> float:
        """Compute the fraction of an axis-aligned box clipped by the half-space n·x < d.

        The box is centred at the origin with half-extents (hx, hy, hz).
        Returns the volume fraction that lies on the *negative* side of the plane
        (i.e. the portion that would be cropped away).
        """
        hx, hy, hz = half_extents
        corners = np.array(
            np.meshgrid(
                [-hx, hx], [-hy, hy], [-hz, hz]
            )
        ).T.reshape(-1, 3)
        proj = corners @ normal
        p_min, p_max = proj.min(), proj.max()

        if d <= p_min:
            return 0.0
        if d >= p_max:
            return 1.0

        n_samples = 128
        xs = np.linspace(-hx, hx, n_samples)
        ys = np.linspace(-hy, hy, n_samples)
        zs = np.linspace(-hz, hz, n_samples)
        grid = np.stack(
            np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        below = (grid @ normal) < d
        return float(below.sum()) / below.size

    @staticmethod
    def cropPoints(
        source_points: np.ndarray,
        crop_ratio: float,
    ) -> np.ndarray:
        if source_points.shape[0] == 0 or crop_ratio <= 0.0:
            return source_points
        if crop_ratio >= 1.0:
            return np.empty((0, source_points.shape[1]), dtype=source_points.dtype)

        bbox_min = source_points.min(axis=0)
        bbox_max = source_points.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        half_extents = (bbox_max - bbox_min) / 2.0

        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        normal = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])

        proj_corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    c = half_extents * np.array([sx, sy, sz])
                    proj_corners.append(c @ normal)
        p_min = min(proj_corners)
        p_max = max(proj_corners)

        lo, hi = p_min, p_max
        for _ in range(64):
            mid = (lo + hi) / 2.0
            frac = PointSampler._clippedBoxVolumeFraction(
                half_extents, normal, mid
            )
            if frac < crop_ratio:
                lo = mid
            else:
                hi = mid
        d = (lo + hi) / 2.0

        proj_pts = (source_points - center) @ normal
        mask = proj_pts >= d
        cropped_points = source_points[mask]
        return cropped_points

    @staticmethod
    def addGaussNoise(
        source_points: np.ndarray,
        noise_ratio: float,
        noise_scale: float,
    ) -> np.ndarray:
        """
        随机选取noise_ratio比例的点云，添加尺度为noise_scale的高斯噪声
        """
        num_points = source_points.shape[0]
        if num_points == 0 or noise_ratio <= 0.0 or noise_scale <= 0.0:
            return source_points.copy()
        num_noisy = int(np.floor(num_points * noise_ratio))
        if num_noisy <= 0:
            return source_points.copy()
        noisy_points = source_points.copy()
        noisy_idx = np.random.choice(num_points, size=min(num_noisy, num_points), replace=False)
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=(noisy_idx.size, source_points.shape[1]))
        noisy_points[noisy_idx] += noise
        return noisy_points

    @staticmethod
    def addDepthSensorNoise(
        source_points: np.ndarray,
        noise_ratio: float,
        noise_scale: float,
    ) -> np.ndarray:
        """Simulate realistic depth-sensor noise on a point cloud.

        Based on the Kinect depth noise model:
            Z' = 35130 / (35130/Z + N(0, sigma_d^2) + 0.5)
        where Z is the original depth and sigma_d = noise_scale.

        For each selected point, the depth along the viewing ray (origin -> point)
        is perturbed according to the formula, producing a displacement along the
        ray direction.
        """
        num_points = source_points.shape[0]
        if num_points == 0 or noise_ratio <= 0.0 or noise_scale <= 0.0:
            return source_points.copy()

        num_noisy = int(np.floor(num_points * noise_ratio))
        if num_noisy <= 0:
            return source_points.copy()

        noisy_points = source_points.copy()

        noisy_idx = np.random.choice(num_points, size=min(num_noisy, num_points), replace=False)
        selected = noisy_points[noisy_idx]

        depths = np.linalg.norm(selected, axis=1)
        valid = depths > 1e-8
        if not np.any(valid):
            return noisy_points

        z = depths[valid]
        disparity = 35130.0 / z
        noisy_disparity = disparity + np.random.normal(0.0, noise_scale, size=z.shape) + 0.5
        noisy_disparity = np.clip(noisy_disparity, 1e-8, None)
        z_noisy = 35130.0 / noisy_disparity

        scale = z_noisy / z
        selected[valid] *= scale[:, np.newaxis]

        noisy_points[noisy_idx] = selected
        return noisy_points

    @staticmethod
    def sampleMeshPoints(
        mesh: trimesh.Trimesh,
        sample_point_num: int,
        sharp_edge_angle: float=30.0,
        sample_edge_ratio: float=0,
        drop_ratio: float=0,
        crop_ratio: float=0,
        gauss_noise_ratio: float=0,
        gauss_noise_scale: float=0,
        depth_sensor_noise_ratio: float=0,
        depth_sensor_noise_scale: float=0,
    ) -> np.ndarray:
        keep_fraction = (1.0 - drop_ratio) * (1.0 - crop_ratio)
        keep_fraction = max(keep_fraction, 0.01)
        total_needed = int(np.ceil(sample_point_num / keep_fraction))

        edge_ratio = np.clip(sample_edge_ratio, 0.0, 1.0)
        num_edge = int(np.round(total_needed * edge_ratio))
        num_surf = total_needed - num_edge

        parts = []
        if num_surf > 0:
            surf_pts = PointSampler.sampleSurfPoints(mesh, num_points=num_surf)
            parts.append(surf_pts)
        if num_edge > 0:
            sharp_edge_pts = PointSampler.sampleSharpEdgePoints(
                mesh, angle_threshold=sharp_edge_angle, num_points=num_edge,
            )
            parts.append(sharp_edge_pts)

        if len(parts) == 0:
            merged_pts = np.empty((0, 3), dtype=np.float64)
        else:
            merged_pts = np.concatenate(parts, axis=0)

        if drop_ratio > 0:
            merged_pts = PointSampler.dropPoints(merged_pts, drop_ratio)

        if crop_ratio > 0:
            merged_pts = PointSampler.cropPoints(merged_pts, crop_ratio)

        if gauss_noise_ratio > 0 and gauss_noise_scale > 0:
            merged_pts = PointSampler.addGaussNoise(
                merged_pts, gauss_noise_ratio, gauss_noise_scale,
            )

        if depth_sensor_noise_ratio > 0 and depth_sensor_noise_scale > 0:
            merged_pts = PointSampler.addDepthSensorNoise(
                merged_pts, depth_sensor_noise_ratio, depth_sensor_noise_scale,
            )

        if merged_pts.shape[0] < sample_point_num:
            deficit = sample_point_num - merged_pts.shape[0]
            fill_pts = PointSampler.sampleSurfPoints(mesh, num_points=deficit)
            merged_pts = np.concatenate([merged_pts, fill_pts], axis=0)

        # 新增：如果点数超出了sample_point_num，则随机采样sample_point_num个点返回
        if merged_pts.shape[0] > sample_point_num:
            idx = np.random.choice(merged_pts.shape[0], sample_point_num, replace=False)
            merged_pts = merged_pts[idx]

        return merged_pts
