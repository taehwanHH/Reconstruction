import numpy as np
from functools import cached_property

from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from trimesh.visual.color import ColorVisuals
import trimesh
import pyvista as pv


class Stiffness:
    def __init__(self, stiff_config):
        self.k_cfg = stiff_config.render.k
        self.k_min, self.k_max, self.k_interval = self.k_cfg.min, self.k_cfg.max, self.k_cfg.interval
    @cached_property
    def k_values(self):
        return np.array(list(range(self.k_min, self.k_max + 1, self.k_interval)))
    @cached_property
    def k_num(self):
        return self.k_values.shape[0]

    def k_normalize(self,k):
        return (k - self.k_values.min()) / (self.k_values.max() - self.k_values.min() + 1e-8)


    def get_local_stiffness(self, positions, bounds):
        """
        positions: (N,3) array of XYZ contact points
        bounds:    (min_bound, max_bound), each length-3
        반환:      (N,) array of stiffness, one per 점
        """

        # 1) z_norm 준비
        z = positions[:, 2]
        z_min, z_max = bounds[0][2], bounds[1][2]
        z_norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)

        # 2) 구획 중심 계산
        centers = (np.arange(self.k_num) + 0.5) / self.k_num  # e.g. [0.1,0.3,...,0.9] for k_num=5

        # 3) 각 z_norm을 가장 가까운 center에 할당
        #    abs(z_norm[:,None] - centers[None,:]) → (N, k_num)
        region_idx = np.argmin(np.abs(z_norm[:, None] - centers[None, :]), axis=1)

        # 4) stiffness 매핑
        return self.k_values[region_idx]


    def show_colored_with_pyvista(
            self,
            mesh: trimesh.Trimesh,
            sample_pts: np.ndarray,  # (N_pts,3)
            sample_ks: np.ndarray  # (N_pts,), normalized to [0,1]
    ):
        # 1) Build KD-tree on your sample cloud
        tree = cKDTree(sample_pts)

        # 2) Query every mesh vertex
        verts = mesh.vertices  # (V,3)
        _, idx = tree.query(verts, k=1)  # idx is length V
        vert_ks = sample_ks[idx]  # (V,)
        print(vert_ks.shape)
        # 3) Build PyVista mesh
        #    Trimesh faces need to be converted into the "offset" format for PyVista:
        faces = mesh.faces  # (F,3)
        # Prepend a 3 to each face to indicate 3 points per polygon:
        pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
        pv_mesh = pv.PolyData(verts, pv_faces)

        # 4) Attach the per-vertex scalars
        pv_mesh.point_data["stiffness"] = vert_ks

        # 5) Plot with a colormap
        plotter = pv.Plotter()
        plotter.add_mesh(
            pv_mesh,
            scalars="stiffness",
            cmap="viridis",
            show_scalar_bar=True,
            lighting=True
        )
        plotter.show()