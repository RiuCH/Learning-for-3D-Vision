

from utils_gif import concat_media_horizontally

# concat_media_horizontally([f'output/eval_point_500.png', f"output/eval_point_1000.png", f"output/eval_point_2000.png"], output_path=f"output/eval_num_points_comparison.gif", 
#                                   captions=["F1 Score for 500 num points", "F1 Score for 1000 num points", "F1 Score for 2000 num points"])




# concat_media_horizontally([f'output/voxel_interpret_0_0.gif', f'output/voxel_interpret_0_1.gif', f'output/voxel_interpret_0_2.gif', f'output/voxel_interpret_0_4.gif'], output_path=f"output/interpret_voxel.gif", 
#                                   captions=["Decoder Layer 1", "Decoder Layer 2", "Decoder Layer 3", "Decoder Layer 4"])



# concat_media_horizontally(['output/0_point.png', f'output/point_prediction_0.gif', f"output/point_1000_prediction_full_0.gif"], output_path=f"output/point_1000_model_0_compare.gif", 
#                                   captions=["Input RGB", "Predicted 3D Points for 1C", "Predicted 3D Points for 3C"])


# concat_media_horizontally(['output/150_point.png', f'output/point_prediction_150.gif', f"output/point_1000_prediction_full_150.gif"], output_path=f"output/point_1000_model_150_compare.gif", 
#                                   captions=["Input RGB", "Predicted 3D Points for 1C", "Predicted 3D Points for 3C"])


concat_media_horizontally(['output/600_point.png', f'output/point_prediction_600.gif', f"output/point_1000_prediction_full_600.gif"], output_path=f"output/point_1000_model_600_compare.gif", 
                                  captions=["Input RGB", "Predicted 3D Points for 1C", "Predicted 3D Points for 3C"])



# concat_media_horizontally([f'output/eval_point_2000.png', f"output/eval_point_1000.png"], output_path=f"output/eval_full_dataset_comparison.gif", 
#                                   captions=["F1 Score for one class training", "F1 Score for three classes training"])

