import pytorch3d
import torch
import numpy as np
import imageio
import os
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from pytorch3d.io import load_obj, load_objs_as_meshes
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh
from code.q3 import get_colorful_textures

def render_spiral_360(path=None,
                image_size=256,
                num_frames=18,
                duration=6,
                device=None,
                output_file="output/render_cow_360.gif"):

    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)

    vertices, faces = load_cow_mesh(path=path)
    vertices = vertices.unsqueeze(0) 
    faces = faces.unsqueeze(0)  
    apply_textures = get_colorful_textures(torch.tensor([1,1,0]), torch.tensor([0,1,1]))
    textures = apply_textures(vertices)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    angles = torch.linspace(-180, 180, num_frames)
    fovs = torch.cat((torch.linspace(60, 120, num_frames//2), torch.linspace(120, 60, num_frames//2)))
    renders = []

    for i, angle in tqdm(enumerate(angles)):
        R, t = pytorch3d.renderer.look_at_view_transform(dist=50 * (abs(angle)/180) + 10, elev=10, azim=angle)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fovs[i], R=R, T=t, device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)    
        draw.text((20, 20), f"Angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop=0)

if __name__ == "__main__":    
    render_spiral_360(path="data/FinalBaseMesh.obj", output_file="output/human.gif", num_frames=10, image_size=256)