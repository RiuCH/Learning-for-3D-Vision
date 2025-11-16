import torch
from code.q1 import render_360
from code.q2 import render_tetrahedon, render_cube
from code.q3 import get_colorful_textures
from code.q4 import do_q4
from code.q5 import create_pc_data, render_pc_360, render_mesh_360, create_torus, create_klein_bottle, create_torus_mesh, create_gyroid_mesh
from code.q6 import render_spiral_360
from code.q7 import sample_points
from starter.dolly_zoom import dolly_zoom

if __name__ == "__main__":
    # Q1
    render_360(path="data/cow.obj", output_file="output/render_cow_360.gif")
    dolly_zoom(output_file="output/dolly.gif")

    # Q2
    render_tetrahedon(output_file="output/tetrahedron.gif")
    render_cube(output_file="output/cube.gif")

    # Q3
    apply_color_texture = get_colorful_textures(torch.tensor([1,1,0]), torch.tensor([0,1,1]))
    render_360(path="data/cow.obj", output_file="output/render_color_cow_360.gif", apply_textures=apply_color_texture)

    # Q4
    do_q4()

    # Q5.1
    pc1, pc2, pc3 = create_pc_data()
    render_pc_360(pc1, output_file="output/pc1.gif")
    render_pc_360(pc2, output_file="output/pc2.gif")
    render_pc_360(pc3, output_file="output/pc3.gif")    

    # Q5.2
    torus_pc = create_torus(num_samples=500)
    render_pc_360(torus_pc, output_file="output/torus.gif")
    klein_pc = create_klein_bottle(num_samples=200)
    render_pc_360(klein_pc, dist=50, output_file="output/klein.gif")

    # Q5.3
    torus_mesh = create_torus_mesh()
    render_mesh_360(torus_mesh, output_file="output/torus_mesh.gif")
    cone_mesh = create_gyroid_mesh()
    render_mesh_360(cone_mesh, output_file="output/gyroid_mesh.gif", dist=20)

    # Q6
    render_spiral_360(path="data/FinalBaseMesh.obj", output_file="output/human.gif", num_frames=10, image_size=256)

    #Q7
    for i in [10, 100, 1000, 10000]:
        cow_pc = sample_points(path='data/cow.obj', num_samples=i)
        render_pc_360(cow_pc, rotate=0, dist=3, num_frames=18, output_file=f"output/sample_cow_{i}.gif")
