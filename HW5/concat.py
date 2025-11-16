from PIL import Image, ImageDraw, ImageFont, ImageSequence
from typing import List, Optional, Tuple


FONT = ImageFont.load_default(20)

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
    


if __name__ == "__main__":

    # concat_media_horizontally(['output/exp_cls_0_10000_0.9790/obj_0_gt_0_pred_0.gif', f"output/exp_cls_0_10000_0.9790/obj_617_gt_1_pred_1.gif", f"output/exp_cls_0_10000_0.9790/obj_719_gt_2_pred_2.gif"], output_path=f"output/exp_cls_0_10000_0.9790/out.gif", 
    #                                   captions=["Class 0: Chair", "Class 1: Vase", "Class 2: Lamp"])



    # concat_media_horizontally(['output/exp_cls_0_10000_0.9790/obj_406_gt_0_pred_2.gif', f"output/exp_cls_0_10000_0.9790/obj_618_gt_1_pred_2.gif", f"output/exp_cls_0_10000_0.9790/obj_827_gt_2_pred_1.gif"], output_path=f"output/exp_cls_0_10000_0.9790/out_fail.gif", 
    #                                   captions=["GT: Chair, Prediction: Lamp", "GT: Vase, Prediction: Lamp", "GT: Lamp, Prediction: Vase"])


    concat_media_horizontally(['output/exp_seg_0_10000_0.9020/obj_0_gt_0.9487.gif', f"output/exp_seg_0_10000_0.9020/obj_0_pred_0.9487.gif"], output_path=f"output/exp_seg_0_10000_0.9020/obj_0.gif", 
                                    captions=["GT", "Prediction"])


    concat_media_horizontally(['output/exp_seg_0_10000_0.9020/obj_1_gt_0.988.gif', f"output/exp_seg_0_10000_0.9020/obj_1_pred_0.988.gif"], output_path=f"output/exp_seg_0_10000_0.9020/obj_1.gif", 
                                    captions=["GT", "Prediction"])


    concat_media_horizontally(['output/exp_seg_0_10000_0.9020/obj_2_gt_0.8983.gif', f"output/exp_seg_0_10000_0.9020/obj_2_pred_0.8983.gif"], output_path=f"output/exp_seg_0_10000_0.9020/obj_2.gif", 
                                    captions=["GT", "Prediction"])


    concat_media_horizontally(['output/exp_seg_0_10000_0.9020/obj_4_gt_0.6845.gif', f"output/exp_seg_0_10000_0.9020/obj_4_pred_0.6845.gif"], output_path=f"output/exp_seg_0_10000_0.9020/obj_3.gif", 
                                    captions=["GT", "Prediction"])


    concat_media_horizontally(['output/exp_seg_0_10000_0.9020/obj_26_gt_0.4824.gif', f"output/exp_seg_0_10000_0.9020/obj_26_pred_0.4824.gif"], output_path=f"output/exp_seg_0_10000_0.9020/obj_4.gif", 
                                    captions=["GT", "Prediction"])

    concat_media_horizontally(['output/exp_seg_0_10000_0.9020/obj_41_gt_0.6632.gif', f"output/exp_seg_0_10000_0.9020/obj_41_pred_0.6632.gif"], output_path=f"output/exp_seg_0_10000_0.9020/obj_5.gif", 
                                    captions=["GT", "Prediction"])