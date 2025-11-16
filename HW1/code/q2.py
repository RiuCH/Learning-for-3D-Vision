import torch
from code.q1 import render_360

def render_tetrahedon(output_file):
    vertices = torch.tensor([[0., 0., 0.], [1., 1., 2.], [2., 0., 0.], [1., 2., 0.]])-1.
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    render_360(vertices=vertices, faces=faces, output_file=output_file)

def render_cube(output_file):
    vertices = torch.tensor([[0., 0., 2.], [2., 0., 2.], [2., 0., 0.], [0., 0., 0.], [2., 2., 0.], 
                             [0., 2., 0.], [0., 2., 2.], [2., 2., 2.]])-1.
    faces = torch.tensor([[0, 2, 1], [0, 3, 2], [2, 3, 5], [2, 5, 4], [6, 5, 0], [0, 5, 3], [1, 6, 0], 
                          [4, 5, 6], [1, 2, 4], [4, 6, 7], [1, 7, 6], [1, 4, 7]])
    render_360(vertices=vertices, faces=faces, output_file=output_file)

if __name__ == "__main__":
    render_tetrahedon(output_file="output/tetrahedron.gif")
    render_cube(output_file="output/cube.gif")

