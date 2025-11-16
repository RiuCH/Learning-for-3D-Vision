import torch
from code.q1 import render_360

def get_colorful_textures(color1, color2):

    def apply_color(vertices):

        z = vertices[:, :, 2][0]
        z_min = z.min()
        z_max = z.max()

        alpha = (z - z_min) / (z_max - z_min)
        alpha = alpha.unsqueeze(-1)
        textures = alpha * color2 + (1 - alpha) * color1
        return textures.unsqueeze(0)
    
    return apply_color

if __name__ == "__main__":
    apply_color_texture = get_colorful_textures(torch.tensor([1,1,0]), torch.tensor([0,1,1]))
    render_360(path="data/cow.obj", output_file="output/render_color_cow_360.gif", apply_textures=apply_color_texture)
