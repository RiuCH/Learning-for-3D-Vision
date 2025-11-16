import torch
import pytorch3d
import torch
import numpy as np
import imageio
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, Pointclouds


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def get_pc_color(points):
    return (points - points.min()) / (points.max() - points.min())

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb


def render_pc_360(
                  pointcloud=None,
                  num_frames=18,
                  image_size=512,
                  duration=3,
                  device=None,
                  dist=5,
                  rotate=0,
                  output_file="output/render_cow_360.gif"):
    
    if device is None:
        device = get_device()

    pointcloud = Pointclouds([pointcloud], features=[get_pc_color(pointcloud)])
    pointcloud = pointcloud.to(device)
        
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
        # draw = ImageDraw.Draw(image)    
        # draw.text((20, 20), f"Angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop=0)


def render_mesh_360(
                  mesh=None,
                  num_frames=18,
                  image_size=512,
                  duration=3,
                  device=None,
                  dist=5,
                  output_file="output/render_cow_360.gif"):
    
    if device is None:
        device = get_device()
    
    textures = (mesh._verts_list[0] - mesh._verts_list[0].min()) / (mesh._verts_list[0].max() - mesh._verts_list[0].min())
    mesh.textures = TexturesVertex([textures])
    
    renderer = get_mesh_renderer(image_size=image_size)

    angles = torch.linspace(-180, 180, num_frames)
    renders = []

    for angle in tqdm(angles):
        R, t = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=0, azim=angle)
        R_ = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, 0, 0]), "XYZ")
        R = R @ R_
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=t, device=device)
        rend = renderer(mesh, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3].clip(0, 1)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        # draw = ImageDraw.Draw(image)    
        # draw.text((20, 20), f"Angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))

    imageio.mimsave(output_file, images, duration=duration, loop=0)
    
def render_voxel_360(
                  voxel=None,
                  num_frames=18,
                  image_size=512,
                  duration=3,
                  device=None,
                  dist=5,
                  output_file="output/render_cow_360.gif"):
    
    if device is None:
        device = get_device()
        
    mesh = cubify(voxel, 0.5)
    textures = (mesh._verts_list[0] - mesh._verts_list[0].min()) / (mesh._verts_list[0].max() - mesh._verts_list[0].min())
    mesh.textures = TexturesVertex([textures])
    
    renderer = get_mesh_renderer(image_size=image_size)

    angles = torch.linspace(-180, 180, num_frames)
    renders = []

    for angle in tqdm(angles):
        R, t = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=0, azim=angle)
        R_ = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, 0, 0]), "XYZ")
        R = R @ R_
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=t, device=device)
        rend = renderer(mesh, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3].clip(0, 1)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        # draw = ImageDraw.Draw(image)    
        # draw.text((20, 20), f"Angle: {angles[i]:.0f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop=0)