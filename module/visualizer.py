import numpy as np
import pyvista as pv
from matplotlib import cm
from pyvistaqt import BackgroundPlotter
import torch
import copy
import matplotlib.pyplot as plt

from os import path as osp
from midastouch.modules.misc import DIRS
from midastouch.modules.particle_filter import Particles
from midastouch.viz.helpers import draw_poses
import queue
from PIL import Image
import tkinter as tk
pv.set_plot_theme("document")


class Viz:
    def __init__(
        self, off_screen: bool = False, zoom: float = 1.0, window_size: int = 0.5
    ):

        pv.global_theme.multi_rendering_splitting_position = 0.5
        """
            subplot(0, 0) main viz
            subplot(0, 1): tactile image viz
            subplot(1, 1): tactile codebook viz 
        """
        shape, row_weights, col_weights = (1, 2), [2], [0.5, 0.5]
        groups = [(np.s_[:], 0), (0, 1)]

        w, h = tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight()

        if off_screen:
            window_size = 1.0
        self.plotter = BackgroundPlotter(
            title="MidasTouch",
            lighting="three lights",
            # window_size=(int(w * window_size), int(h * window_size)),
            window_size=(900, 500),
            off_screen=off_screen,
            shape=shape,
            row_weights=row_weights,
            col_weights=col_weights,
            groups=groups,
            border_color="white",
            toolbar=False,
            menu_bar=False,
            auto_update=True,
        )
        self.zoom = zoom

        self.viz_queue = queue.Queue(1)
        self.plotter.add_callback(self.update_viz, interval=500)
        self.pause = False
        self.font_size = int(30 * window_size)
        self.off_screen = off_screen

    def toggle_vis(self, flag):
        self.mesh_actor.SetVisibility(flag)

    def pause_vis(self, flag):
        self.pause = flag

    def set_camera(self, position="yz", azimuth=45, elevation=20, zoom=None):
        (
            self.plotter.camera_position,
            self.plotter.camera.azimuth,
            self.plotter.camera.elevation,
        ) = (position, azimuth, elevation)
        if zoom is None:
            self.plotter.camera.Zoom(self.zoom)
        else:
            self.plotter.camera.Zoom(zoom)
        self.plotter.camera_set = True

    def mirror_view(self):
        self.plotter.subplot(0, 0)
        cam = self.plotter.camera.copy()
        self.plotter.subplot(1, 1)
        self.plotter.camera = cam
        self.plotter.camera.Zoom(0.8)

    def reset_vis(self, flag):
        self.plotter.subplot(0, 0)
        self.set_camera()
        self.reset_widget.value = not flag

    def init_variables(
        self,
        obj_model: str,
        mesh_path: str,
        frame_rate: int = 30,
    ):
        self.mesh_pv = pv.read(mesh_path)  # pyvista object
        self.mesh_pv_deci = pv.read(
            mesh_path.replace("nontextured", "nontextured_decimated")
        )  # decimated pyvista object
        self.frame_rate = frame_rate


        # Filter window
        self.plotter.subplot(0, 0)
        dargs = dict(
            color="grey",
            ambient=0.6,
            opacity=0.5,
            smooth_shading=True,
            specular=1.0,
            show_scalar_bar=False,
            render=False,
        )
        self.mesh_actor = self.plotter.add_mesh(self.mesh_pv, **dargs)

        if not self.off_screen:
            pos, offset = self.plotter.window_size[1] - 40, 10
            widget_size = 25
            # self.plotter.add_checkbox_button_widget(
            #     self.toggle_vis,
            #     value=True,
            #     color_off="white",
            #     color_on="black",
            #     position=(10, pos),
            #     size=widget_size,
            # )
            # self.plotter.add_text(
            #     "Toggle object",
            #     position=(15 + widget_size, pos),
            #     color="black",
            #     font="times",
            #     font_size=self.font_size,
            # )
            # self.reset_widget = self.plotter.add_checkbox_button_widget(
            #     self.reset_vis,
            #     value=True,
            #     color_off="white",
            #     color_on="white",
            #     background_color="gray",
            #     position=(10, pos - (widget_size + offset)),
            #     size=widget_size,
            # )
            # self.plotter.add_text(
            #     "Reset camera",
            #     position=(15 + widget_size, pos - (widget_size + offset)),
            #     color="black",
            #     font="times",
            #     font_size=self.font_size,
            # )
            # self.plotter.add_checkbox_button_widget(
            #     self.pause_vis,
            #     value=False,
            #     color_off="white",
            #     color_on="black",
            #     position=(10, pos - 2 * (widget_size + offset)),
            #     size=widget_size,
            # )
            # self.plotter.add_text(
            #     "Pause",
            #     position=(15 + widget_size, pos - 2 * (widget_size + offset)),
            #     color="black",
            #     font="times",
            #     font_size=self.font_size,
            # )
        self.set_camera()

        self.max_clusters = 5
        self.moving_sigma = self.max_clusters * [None]
        for i in range(self.max_clusters):
            self.moving_sigma[i] = pv.ParametricEllipsoid(0.0, 0.0, 0.0)
            dargs = dict(
                color="red",
                ambient=0.0,
                opacity=0.2,
                smooth_shading=True,
                show_edges=False,
                specular=1.0,
                show_scalar_bar=False,
                render=False,
            )
            self.plotter.add_mesh(self.moving_sigma[i], **dargs)

        dargs = dict(
            color="tan",
            ambient=0.0,
            opacity=0.7,
            smooth_shading=True,
            show_edges=False,
            specular=1.0,
            show_scalar_bar=False,
            render=False,
        )

        dargs = dict(
            color="red",
            show_scalar_bar=False,
            opacity=0.3,
            reset_camera=False,
            render=False,
        )

        self.plotter.add_text(
            "Contact point",
            position="bottom",
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="Object text",
        )

        # Tactile window
        self.plotter.subplot(0, 1)
        self.plotter.camera.Zoom(1)
        self.plotter.add_text(
            "Tactile image and heightmap",
            position="bottom",
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="Tactile text",
        )

        # # Heatmap window
        # self.plotter.subplot(1, 1)
        # dargs = dict(
        #     color="tan",
        #     ambient=0.0,
        #     opacity=0.7,
        #     smooth_shading=True,
        #     show_edges=False,
        #     specular=1.0,
        #     show_scalar_bar=False,
        #     render=False,
        # )

        self.image_plane, self.heightmap_plane = None, None
        self.images = {"im": [], "path": []}
        self.image_plane, self.heightmap_plane = None, None


    def update_viz(
            self,
    ):
        if self.viz_queue.qsize():
            (
                sampled_points,
                contact_masks,
                heightmaps,
                tactile_images,
                frame,
                image_savepath
            ) = self.viz_queue.get()
            self.viz_contact(sampled_points,frame)
            self.viz_tactile_image(tactile_images,heightmaps,contact_masks)
            # self.mirror_view()
            self.plotter.add_text(
                f"\nFrame {frame}   ",
                position="upper_right",
                color="black",
                shadow=True,
                font="times",
                font_size=self.font_size,
                name="frame text",
                render=True,
            )
            if image_savepath:
                self.images["im"].append(self.plotter.screenshot())
                self.images["path"].append(image_savepath)
            self.viz_queue.task_done()

    def update(
            self,
            sampled_points: np.ndarray = None,
            contact_masks: np.ndarray = None,
            heightmaps: np.ndarray = None,
            tactile_images: np.ndarray = None,
            frame: int = None,
            image_path: str = None
    ) -> None:
        """
        메쉬와 샘플링된 접촉점, 센서 포즈, 접촉 마스크, 높이 맵, 촉각 이미지를 업데이트.

        :param sampled_points: 샘플링된 접촉점 좌표 (N, 3 크기의 numpy 배열).
        :param sensor_poses: 센서 포즈 (N, 4x4 변환 행렬의 numpy 배열).
        :param contact_masks: 접촉 마스크 데이터.
        :param heightmaps: 높이 맵 데이터.
        :param tactile_images: 촉각 이미지 데이터.
        :param frame: 현재 프레임 번호.
        """
        if self.viz_queue.full():
            self.viz_queue.get()

        # 큐에 필요한 데이터를 추가
        self.viz_queue.put(
            (
                sampled_points,
                contact_masks,
                heightmaps,
                tactile_images,
                frame,
                image_path
            ),
            block=False,
        )

    def viz_contact(self, contact_points: np.ndarray, frame: int) -> None:
        self.plotter.subplot(0, 0)

        self.plotter.add_mesh(self.mesh_pv, color="white", opacity=0.5)
        self.plotter.add_points(contact_points, color="red", point_size=5)
        # self.plotter.add_text(f"Frame: {frame}", position="upper_left", font_size=12)
        # self.plotter.render()

    def viz_tactile_image(
        self,
        image: np.ndarray,
        heightmap: torch.Tensor,
        mask: torch.Tensor,
        s: float = 1.8e-3 ,
    ) -> None:
        if self.image_plane is None:
            self.image_plane = pv.Plane(
                i_size=image.shape[1] * s,
                j_size=image.shape[0] * s,
                i_resolution=image.shape[1] - 1,
                j_resolution=image.shape[0] - 1,
            )
            self.image_plane.points[:, -1] = 0.25
            self.heightmap_plane = copy.deepcopy(self.image_plane)

        # # visualize gelsight image
        # # 촉각 이미지 시각화 (subplot(1,0))
        # self.plotter.subplot(0, 1)
        # image_tex = pv.numpy_to_texture(image)
        # self.plotter.add_mesh(
        #     self.image_plane,
        #     texture=image_tex,
        #     smooth_shading=False,
        #     show_scalar_bar=False,
        #     name="image",
        #     render=False,
        # )
        # self.plotter.add_text(
        #     "Tactile Image",
        #     position="bottom",
        #     color="black",
        #     shadow=True,
        #     font="times",
        #     font_size=self.font_size,
        #     name="Tactile text",
        # )
        #
        # # 높이 맵 시각화 (subplot(1,1))
        # self.plotter.subplot(1, 1)
        # heightmap, mask = heightmap.cpu().numpy(), mask.cpu().numpy()
        # heightmap_tex = pv.numpy_to_texture(-heightmap * mask.astype(np.float32))
        # self.heightmap_plane.points[:, -1] = (
        #         np.flip(heightmap * mask.astype(np.float32), axis=0).ravel() * (0.5 * s)
        #         - 0.15
        # )
        # self.plotter.add_mesh(
        #     self.heightmap_plane,
        #     texture=heightmap_tex,
        #     cmap=cm.get_cmap("plasma"),
        #     show_scalar_bar=False,
        #     name="heightmap",
        #     render=False,
        # )
        # self.plotter.add_text(
        #     "Heightmap",
        #     position="bottom",
        #     color="black",
        #     shadow=True,
        #     font="times",
        #     font_size=self.font_size,
        #     name="Heightmap text",
        # )
        self.plotter.subplot(0, 1)

        heightmap, mask = heightmap.cpu().numpy(), mask.cpu().numpy()
        image_tex = pv.numpy_to_texture(image)

        heightmap_tex = pv.numpy_to_texture(-heightmap * mask.astype(np.float32))
        self.heightmap_plane.points[:, -1] = (
            np.flip(heightmap * mask.astype(np.float32), axis=0).ravel() * (0.5 * s)
            - 0.15
        )
        self.plotter.add_mesh(
            self.image_plane,
            texture=image_tex,
            smooth_shading=False,
            show_scalar_bar=False,
            name="image",
            render=False,
        )

        self.plotter.add_mesh(
            self.heightmap_plane,
            texture=heightmap_tex,
            cmap=cm.get_cmap("plasma"),
            show_scalar_bar=False,
            name="heightmap",
            render=False,
        )

    def close(self):
        if len(self.images):
            for (im, path) in zip(self.images["im"], self.images["path"]):
                im = Image.fromarray(im.astype("uint8"), "RGB")
                im.save(path)

        self.plotter.close()