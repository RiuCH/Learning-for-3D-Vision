import pytorch3d
import torch
import numpy as np
import imageio
import os
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh
from starter.dolly_zoom import dolly_zoom

def render_360(path=None,
                vertices=None, 
                faces=None,
                apply_textures=None,
                image_size=256,
                color=[0.7, 0.7, 1],
                num_frames=18,
                duration=6,
                device=None,
                output_file="output/render_cow_360.gif"):

    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)

    if path is not None:
        vertices, faces = load_cow_mesh(path=path)
    elif vertices is not None and faces is not None:
        pass
    else:
        raise ValueError("Either path or vertices and faces must be specified.")

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    if apply_textures is None:
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor(color)  # (1, N_v, 3)
    else:
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
    renders = []

    for angle in tqdm(angles):
        R, t = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=angle)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=t, device=device)

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
    render_360(path="data/cow.obj", output_file="output/render_cow_360.gif")
    dolly_zoom()