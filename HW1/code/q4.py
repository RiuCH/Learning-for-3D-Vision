import torch
import pytorch3d
import numpy as np
import matplotlib.pyplot as plt
from code.q1 import render_360
from starter.camera_transforms import render_textured_cow

def do_q4():
    img1 = render_textured_cow(R_relative=pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, 0, -np.pi/2]), "XYZ"), T_relative=[0, 0, 0])
    plt.imsave("output/q4_1.jpg", img1)
    img2 = render_textured_cow(R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], T_relative=[0, 0, 2])
    plt.imsave("output/q4_2.jpg", img2)
    img3 = render_textured_cow(R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], T_relative=[1, 0, 0])
    plt.imsave("output/q4_3.jpg", img3)
    img4 = render_textured_cow(R_relative=pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, np.pi/2, 0]), "XYZ"), T_relative=[-3, 0, 3])
    plt.imsave("output/q4_4.jpg", img4)

if __name__ == "__main__":
    do_q4()

