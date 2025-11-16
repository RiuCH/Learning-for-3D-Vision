import pytorch3d
import torch
from PIL import Image
from starter.utils import load_cow_mesh
from code.q5 import render_pc_360

def sample_points(path='data/cow.obj', num_samples=1000):
    vertices, faces = load_cow_mesh(path)
    face_verts = vertices[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    
    face_areas = torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1) / 2.0
    
    face_probabilities = face_areas / face_areas.sum()
   
    sampled_face_indices = torch.multinomial(face_probabilities, num_samples, replacement=True)
    
    rand1 = torch.rand(num_samples, 1, device=vertices.device)
    rand2 = torch.rand(num_samples, 1, device=vertices.device)
    
    sqrt_r1 = torch.sqrt(rand1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - rand2)
    w = sqrt_r1 * rand2
    
    barycentric_coords = torch.cat([u, v, w], dim=1)
    
    sampled_face_verts = face_verts[sampled_face_indices]
    
    points = torch.einsum("ni,nij->nj", barycentric_coords, sampled_face_verts)
    color = (points - points.min()) / (points.max() - points.min())

    point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color])
    
    return point_cloud

def concat_gifs_horizontally(gif_paths, output_path):
    gifs = [Image.open(path) for path in gif_paths]

    num_frames = gifs[0].n_frames
    concatenated_frames = []

    for i in range(num_frames):
        frame_width = sum(gif.size[0] for gif in gifs)
        frame_height = max(gif.size[1] for gif in gifs)
        new_frame = Image.new('RGBA', (frame_width, frame_height))

        x_offset = 0
        for gif in gifs:
            gif.seek(i)  
            new_frame.paste(gif.convert('RGBA'), (x_offset, 0))
            x_offset += gif.size[0]

        concatenated_frames.append(new_frame)
    concatenated_frames[0].save(
        output_path,
        save_all=True,
        append_images=concatenated_frames[1:],
        duration=gifs[0].info['duration'],
        loop=0  
    )
    print(f"Successfully concatenated {len(gif_paths)} GIFs and saved to {output_path}")

if __name__ == "__main__":    

    for i in [10, 100, 1000, 10000]:
        cow_pc = sample_points(path='data/cow.obj', num_samples=i)
        render_pc_360(cow_pc, rotate=0, dist=3, num_frames=18, output_file=f"output/sample_cow_{i}.gif")
    
    concat_gifs_horizontally([f"output/sample_cow_{i}.gif" for i in [10, 100, 1000, 10000]] + ["output/render_cow_360.gif"], "output/sample_cow.gif")