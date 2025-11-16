from PIL import Image

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

# gif_paths = ["images/part_7_a5.0_b0.05.gif", "images/part_7_a10.0_b0.05.gif", "images/part_7_a20.0_b0.05.gif"]
# output_path = "images/part_7_alpha.gif"
# concat_gifs_horizontally(gif_paths, output_path)

gif_paths = ["images/part_7_geometry_a5.0_b0.05.gif", "images/part_7_geometry_a10.0_b0.05.gif", "images/part_7_geometry_a20.0_b0.05.gif"]
output_path = "images/part_7_alpha_geo.gif"
concat_gifs_horizontally(gif_paths, output_path)

gif_paths = ["images/part_7_a10.0_b0.05.gif", "images/part_7_a10.0_b0.1.gif"]
output_path = "images/part_7_beta.gif"
concat_gifs_horizontally(gif_paths, output_path)

gif_paths = [ "images/part_7_geometry_a10.0_b0.05.gif", "images/part_7_geometry_a10.0_b0.1.gif"]
output_path = "images/part_7_beta_geo.gif"
concat_gifs_horizontally(gif_paths, output_path)