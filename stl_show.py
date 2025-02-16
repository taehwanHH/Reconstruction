import pyvista as pv
import trimesh
from midastouch.modules.misc import images_to_video
# 🔹 STL 파일 불러오기
stl_file = "obj_models/011_banana/nontextured.stl"

stl_file2 = "STL/digit.STL"
# stl_file = "STL/digit.STL"
# stl_file2 = "STL/gel_surface_c_mm.obj"

obj_name_name = "017_orange"

# stl_file = "data/sim/011_banana/full_coverage/reconstructed_mesh_poisson.stl"
# stl_file = "data/sim/011_banana/full_coverage/reconstructed_mesh.stl"
mesh = pv.read(stl_file)
mesh2 = pv.read(stl_file2)
# mesh2 = pv.read(stl_file2)
tri_mesh = trimesh.load_mesh(stl_file)
tri_mesh2 = trimesh.load_mesh(stl_file2)
# tri_mesh2 = trimesh.load_mesh(stl_file2)
# 🔹 GLB 파일로 저장 (PowerPoint에서 사용 가능)
# glb_file = "/banana_nontextured.glb"
# tri_mesh.export(glb_file)

# 🔹 시각화

# jpg_file = "data/sim/011_banana/full_coverage/sensed_result"

# images_to_video(jpg_file)

dargs = dict(
    color="grey",
    ambient=0.5,
    opacity=1,
    smooth_shading=True,

    specular=1.0,
    show_scalar_bar=False,
    render=False,
)


plotter = pv.Plotter()

plotter.add_mesh(mesh, **dargs)

plotter.add_mesh(mesh2, **dargs)

plotter.add_axes()
plotter.show(title="STL Model Viewer")
