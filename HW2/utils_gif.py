from PIL import Image, ImageDraw, ImageFont, ImageSequence
from typing import List, Optional, Tuple


FONT = ImageFont.load_default(30)

CAPTION_HEIGHT = 50

def get_frames(path: str, duration: Optional[int] = None) -> List[Image.Image]:
    img = Image.open(path)
    frames = []

    if img.is_animated:
        for frame in ImageSequence.Iterator(img):
            frames.append(frame.convert('RGBA'))
    else:
        frame = img.convert('RGBA')
        frames.append(frame)

    return frames, img.info.get('duration', duration or 100)

def concat_media_horizontally(
    media_paths: List[str],
    output_path: str,
    captions: Optional[List[str]] = None,
    default_duration: int = 3
):
    if captions is None:
        captions = [''] * len(media_paths)
    elif len(captions) != len(media_paths):
        raise ValueError("The number of media paths and captions must be the same.")

    all_media_frames_and_durations = [get_frames(path, default_duration) for path in media_paths]
    
    media_frames = [item[0] for item in all_media_frames_and_durations]
    
    base_durations = [item[1] for item in all_media_frames_and_durations]
    
    num_elements = len(media_frames)
    
    max_frames = max(len(frames) for frames in media_frames)

    gif_durations = [d for d, frames in zip(base_durations, media_frames) if len(frames) > 1]
    final_duration = min(gif_durations) if gif_durations else default_duration

    concatenated_frames = []

    for i in range(max_frames):
        frame_width_no_caption = sum(frames[i % len(frames)].size[0] for frames in media_frames)
        frame_height_no_caption = max(frames[i % len(frames)].size[1] for frames in media_frames)
        
        final_frame_width = frame_width_no_caption
        final_frame_height = frame_height_no_caption + CAPTION_HEIGHT
        
        new_frame = Image.new('RGBA', (final_frame_width, final_frame_height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(new_frame)

        x_offset = 0
        for j in range(num_elements):
            current_frame = media_frames[j][i % len(media_frames[j])]

            new_frame.paste(current_frame, (x_offset, 0))

            caption = captions[j]
            if caption:
                bbox = draw.textbbox((0,0), caption, font=FONT)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x_offset + (current_frame.size[0] - text_width) // 2
                text_y = frame_height_no_caption + (CAPTION_HEIGHT - text_height) // 2
                
                draw.text((text_x + 1, text_y + 1), caption, font=FONT, fill=(0, 0, 0, 100))
                draw.text((text_x, text_y), caption, font=FONT, fill=(0, 0, 0))

            x_offset += current_frame.size[0]

        concatenated_frames.append(new_frame)

    concatenated_frames[0].save(
        output_path,
        save_all=True,
        append_images=concatenated_frames[1:],
        duration=final_duration,
        loop=0
    )
    print(f"Successfully concatenated {num_elements} media elements and saved to {output_path} with {max_frames} frames.")