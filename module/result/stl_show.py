import pyvista as pv
import numpy as np
from PIL import Image
import io
import trimesh.transformations as tf
import trimesh
import os.path as osp
from midastouch.modules.misc import images_to_video
# 🔹 STL 파일 불러오기
obj = "011_banana"
num_samples = 1000
# stl_file = "../../obj_models/011_banana/nontextured.stl"
# stl_file = f"../../data/sim/results/scheme2/recon_stl/{num_samples}_re_mesh.stl"
# stl_file = "../../data/sim/results/num_sample_sweep/scheme1/recon_stl/scheme1_200_re_mesh.stl"
# stl_file = f"../../sim_main_result/recon_stl/{num_samples}_re_mesh.stl"
# stl_file = f"../../data/sim/recon_stl/{num_samples}_re_mesh.stl"

# stl_file2 = "STL/digit.STL"
# stl_file = "STL/digit.STL"
# stl_file2 = "STL/gel_surface_c_mm.obj"

# obj_name_name = "017_orange"

# stl_file = "data/sim/011_banana/full_coverage/reconstructed_mesh_poisson.stl"
# stl_file = "data/sim/011_banana/full_coverage/reconstructed_mesh.stl"
# mesh = pv.read(stl_file)
# mesh2 = pv.read(stl_file2)
# mesh2 = pv.read(stl_file2)
# tri_mesh = trimesh.load_mesh(stl_file)
# tri_mesh2 = trimesh.load_mesh(stl_file2)
# tri_mesh2 = trimesh.load_mesh(stl_file2)
# 🔹 GLB 파일로 저장 (PowerPoint에서 사용 가능)

# tri_mesh.visual_vertex_colors=[255,255,0,255]
# glb_file = f"../../sim_main_result/AE_{num_samples}.glb"
# glb_file = f"../../sim_main_result/local_{num_samples}.glb"

# # tri_mesh.export(glb_file)
#
# rot_matrix = tf.rotation_matrix(np.pi , [1, 1, 0], point=tri_mesh.centroid)
#
# tri_mesh.apply_transform(rot_matrix)
# scene = trimesh.Scene(tri_mesh)
# # 명시적으로 카메라 설정 업데이트
# scene.camera.resolution = (512,512)
# scene.camera.fov = [60, 60]  # 예: 수평, 수직 시야각
#
# png_bytes = scene.save_image(resolution=(512,512), visible=True)
# if png_bytes is None:
#     raise RuntimeError("Rendering failed. Check your OpenGL/pyglet settings.")
# image = Image.open(io.BytesIO(png_bytes))
#
# # img_file = f"../../sim_main_result/AE_{num_samples}.png"
# img_file = f"../../sim_main_result/local_{num_samples}.png"

fn = "../../data/sim/results/num_sample_sweep/Scheme2/recon_stl/Scheme2_500_re_mesh.ply"
mesh = trimesh.load(fn)

mesh.show()

# 🔹 시각화

# jpg_file = "data/sim/011_banana/full_coverage/sensed_result"

# images_to_video(jpg_file)
#
# dargs = dict(
#     color="grey",
#     ambient=0.5,
#     opacity=1,
#     smooth_shading=True,
#
#     specular=1.0,
#     show_scalar_bar=False,
#     render=False,
# )
#
#
# plotter = pv.Plotter()
#
# plotter.add_mesh(mesh, **dargs)
#
# # plotter.add_mesh(mesh2, **dargs)
#
# plotter.add_axes()
# plotter.show(title="STL Model Viewer")
