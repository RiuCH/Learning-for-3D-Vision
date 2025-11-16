import pytorch3d
import torch
import numpy as np
import imageio
import os
import mcubes
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from starter.utils import get_device, unproject_depth_image, get_points_renderer, get_mesh_renderer
from starter.render_generic import load_rgbd_data
import matplotlib.pyplot as plt

def create_pc_data():
    data = load_rgbd_data()
    points1, rgb1 = unproject_depth_image(torch.from_numpy(data["rgb1"]), torch.from_numpy(data["mask1"]), torch.from_numpy(data["depth1"]), data["cameras1"])
    points2, rgb2 = unproject_depth_image(torch.from_numpy(data["rgb2"]), torch.from_numpy(data["mask2"]), torch.from_numpy(data["depth2"]), data["cameras2"])
    pc1, pc2 = pytorch3d.structures.Pointclouds(points=[points1], features=[rgb1]), pytorch3d.structures.Pointclouds(points=[points2], features=[rgb2])
    pc3 = pytorch3d.structures.Pointclouds(points=[torch.vstack([points1, points2])], features=[torch.vstack([rgb1, rgb2])])
    return pc1, pc2, pc3

def render_pc_360(
                  pointcloud=None,
                  num_frames=9,
                  image_size=256,
                  duration=3,
                  device=None,
                  dist=5,
                  rotate=np.pi,
                  output_file="output/render_cow_360.gif"):
    
    if device is None:
        device = get_device()
    
    renderer = get_points_renderer(image_size=image_size)

    angles = torch.linspace(-180, 180, num_frames)
    renders = []

    for angle in tqdm(angles):
        R, t = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=0, azim=angle)
        R_ = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, 0, rotate]), "XYZ")
        R = R @ R_
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=t, device=device)

        rend = renderer(point_clouds=pointcloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3].clip(0, 1) 
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)    
        draw.text((20, 20), f"Angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop=0)


def render_mesh_360(
                  mesh=None,
                  num_frames=9,
                  image_size=256,
                  duration=3,
                  device=None,
                  dist=5,
                  output_file="output/render_cow_360.gif"):
    
    if device is None:
        device = get_device()
    
    renderer = get_mesh_renderer(image_size=image_size)

    angles = torch.linspace(-180, 180, num_frames)
    renders = []

    for angle in tqdm(angles):
        R, t = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=0, azim=angle)
        R_ = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, 0, np.pi]), "XYZ")
        R = R @ R_
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=t, device=device)
        rend = renderer(mesh, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3].clip(0, 1)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)    
        draw.text((20, 20), f"Angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))

    imageio.mimsave(output_file, images, duration=duration, loop=0)


def create_torus(num_samples=200, device=None):
    """
    Create a torus pointcloud using parametric sampling. Samples num_samples ** 2 points.
    """
    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    R = 1
    r = 0.5
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (R + r * torch.cos(Theta))* torch.cos(Phi) 
    y = (R + r * torch.cos(Theta))* torch.sin(Phi) 
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    return point_cloud

def create_klein_bottle(num_samples=200, device=None):
    """
    Create a klein bottle pointcloud using parametric sampling. Samples num_samples ** 2 points.
    """
    if device is None:
        device = get_device()

    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(0, 2 * np.pi, num_samples)

    # Densely sample phi and theta on a grid
    U, V = torch.meshgrid(u, v)

    half = (0 <= U) & (U < np.pi)
    r = 4*(1 - torch.cos(U)/2)
    x = 6*torch.cos(U)*(1 + torch.sin(U)) + r*torch.cos(V + np.pi)
    x[half] = (
        (6*torch.cos(U)*(1 + torch.sin(U)) + r*torch.cos(U)*torch.cos(V))[half])
    y = 16 * torch.sin(U)
    y[half] = (16*torch.sin(U) + r*torch.sin(U)*torch.cos(V))[half]
    z = r * torch.sin(V)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    return point_cloud

def create_torus_mesh(voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -2.0
    max_value = 2.0
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    R = 1.0
    r = 0.5
    voxels = (torch.sqrt(X ** 2 + Y ** 2) - R) ** 2 + Z ** 2 - r ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    return mesh

def create_gyroid_mesh(voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -2 * np.pi
    max_value =  2 * np.pi
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = torch.sin(X) * torch.cos(Y) + torch.sin(Y) * torch.cos(Z) + torch.sin(Z) * torch.cos(X)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    return mesh

if __name__ == "__main__":    
    pc1, pc2, pc3 = create_pc_data()
    render_pc_360(pc1, output_file="output/pc1.gif")
    render_pc_360(pc2, output_file="output/pc2.gif")
    # render_pc_360(pc3, output_file="output/pc3.gif")    

    torus_pc = create_torus(num_samples=500)
    render_pc_360(torus_pc, output_file="output/torus.gif")

    klein_pc = create_klein_bottle(num_samples=200)
    render_pc_360(klein_pc, dist=50, output_file="output/klein.gif")

    torus_mesh = create_torus_mesh()
    render_mesh_360(torus_mesh, output_file="output/torus_mesh.gif")

    cone_mesh = create_gyroid_mesh()
    render_mesh_360(cone_mesh, output_file="output/gyroid_mesh.gif", dist=20)
